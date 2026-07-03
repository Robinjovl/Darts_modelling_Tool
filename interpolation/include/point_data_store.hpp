#ifndef DARTS_POINT_DATA_STORE_HPP
#define DARTS_POINT_DATA_STORE_HPP

// ============================================================================
// point_data_store — hybrid supporting-point cache.
//
// Drop-in replacement for the adaptive interpolators'
//   std::unordered_map<cell_key_t<N_DIMS>, std::array<value_t,N_OPS>, cell_key_hash<N_DIMS>>
// point_data member. It reproduces EXACTLY the map surface the interpolator hot
// path uses (find/at/operator[]/emplace/clear/reserve/size/begin/end with
// it->first / it->second), so the .tpp/.cu code changes by a single member-type
// token per branch.
//
// Storage is a HYBRID:
//   * arena_  — an mmap'd, IMMUTABLE open-addressing hash table (the on-disk cache
//               base). Loaded in O(1) by mmap (no per-point rebuild) and used in
//               place for lookups; its pages are file-backed / demand-paged, NOT
//               anonymous RAM -> the ~150 GB resident-map OOM is structurally gone.
//   * overlay_ — a small in-RAM std::unordered_map holding ONLY the points
//               materialized THIS run (since the last compaction).
//
// Lookups consult overlay_ first, then probe arena_. INSERTS ARE OVERLAY-ONLY
// (emplace/operator[] never touch the arena): every insert site in the
// interpolators is preceded by a full find() (overlay+arena) miss, so a key that
// already lives in the arena is never re-inserted, and the overlay therefore
// never shadows the arena with a duplicate on the normal paths. size() returns
// overlay_.size()+arena_.count (an over-count only by |overlay∩arena|, which is 0
// on the normal paths; every consumer tolerates the bound). The exact deduped
// count is computed only at compaction (build_arena_file).
//
// Concurrency: the only concurrent reader of point_data is the interpolator's
// Phase-2c parallel hypercube assembly (point_data.at() under #pragma omp parallel
// for); at that point both layers are immutable (arena is const mmap; the last
// overlay insert was the *serial* Phase-2b loop, ordered before the parallel
// region by its entry barrier). find()/at()/probe() are pure const reads -> data
// race free with no added locking.
//
// value_t is double for every compiled interpolator; the arena vals region is
// contiguous little-endian float64, 8-byte aligned (page-aligned region start), so
// at()/val_at() return a const std::array<value_t,N_OPS>& by reinterpret_cast
// directly into the mmap (zero copy). Endianness/alignment are recorded in the
// cache-file header and validated by the Python loader before mmap_arena_at is called.
// Cross-platform: both POSIX mmap (Linux/macOS) and the Win32
// CreateFileMapping/MapViewOfFile path are implemented (see map_readonly /
// map_rw_new / flush_unmap_rw) and the cache-file bytes are identical on every OS,
// so the arena is used on all supported platforms. The Python loader
// (darts/tools/obl_cache.py) selects the arena path on interpolator capability
// (has_arena / mmap_arena), never on the OS; interpolators built without arena
// support (static/non-adaptive itors, some GPU builds) use the plain-pickle cache.
// ============================================================================

#include <array>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "multi_index_key.hpp"

#if defined(_WIN32)
// <windows.h> otherwise defines min()/max() as function-like macros that clobber
// std::max, std::numeric_limits<>::max() and Eigen/valarray .max() in any header
// compiled after this one (e.g. interpolation/include/matrix.h). NOMINMAX suppresses
// them; WIN32_LEAN_AND_MEAN trims the include surface (and the rpcndr small/interface
// macros). Guarded so we never redefine a value the build already set.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>      // _fileno, _get_osfhandle
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace pds_detail
{
// High 64 bits of a 64x64 product (portable across GCC/Clang/MSVC).
static inline uint64_t mulhi64(uint64_t a, uint64_t b)
{
#if defined(__SIZEOF_INT128__)
  return static_cast<uint64_t>((static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b)) >> 64);
#elif defined(_MSC_VER) && defined(_M_X64)
  return __umulh(a, b);
#else
  const uint64_t al = a & 0xffffffffull, ah = a >> 32;
  const uint64_t bl = b & 0xffffffffull, bh = b >> 32;
  const uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
  const uint64_t cross = (ll >> 32) + (lh & 0xffffffffull) + (hl & 0xffffffffull);
  return hh + (lh >> 32) + (hl >> 32) + (cross >> 32);
#endif
}
// Lemire reduction: map a uniform 64-bit hash to [0, C) WITHOUT a modulo, so the arena
// capacity C can be any integer (not just a power of two). This lets the table run at a
// tight load factor instead of the ~2x over-provisioning a power-of-two table forces,
// shrinking the on-disk arena (and thus the build/migration write cost) by ~1/3.
static inline size_t home_slot(uint64_t h, size_t C)
{
  return static_cast<size_t>(mulhi64(h, static_cast<uint64_t>(C)));
}

// ----------------------------------------------------------------------------
// Cross-platform memory-mapping + file primitives. The arena file format is the
// SAME bytes on every OS; only these few primitives differ (POSIX mmap vs Win32
// CreateFileMapping/MapViewOfFile). Everything else (build/load/iterate) is shared.
// ----------------------------------------------------------------------------

// Available physical RAM in bytes (0 = unknown -> caller uses the memory-bounded builder).
static inline size_t available_ram_bytes()
{
#if defined(_WIN32)
  MEMORYSTATUSEX ms;
  ms.dwLength = sizeof(ms);
  if (::GlobalMemoryStatusEx(&ms))
    return static_cast<size_t>(ms.ullAvailPhys);
  return 0;
#elif defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
  const long pages = ::sysconf(_SC_AVPHYS_PAGES);
  const long psz = ::sysconf(_SC_PAGESIZE);
  if (pages > 0 && psz > 0)
    return static_cast<size_t>(pages) * static_cast<size_t>(psz);
  return 0;
#else
  return 0;
#endif
}

// Flush a stdio stream's data to stable storage (durability before the atomic rename).
static inline void fsync_file(std::FILE *f)
{
  std::fflush(f);
#if defined(_WIN32)
  const intptr_t h = ::_get_osfhandle(::_fileno(f));
  if (h != -1)
    ::FlushFileBuffers(reinterpret_cast<HANDLE>(h));
#else
  ::fsync(::fileno(f));
#endif
}

// Read-only whole-file mapping (the immutable arena base). Returns base (nullptr on
// failure) and sets *out_len to the file length. The OS keeps the file open until the
// view is released (POSIX: fd closed but mapping holds it; Win32: documented behaviour).
static inline void *map_readonly(const char *path, size_t *out_len)
{
#if defined(_WIN32)
  HANDLE hf = ::CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (hf == INVALID_HANDLE_VALUE)
    return nullptr;
  LARGE_INTEGER sz;
  if (!::GetFileSizeEx(hf, &sz))
  {
    ::CloseHandle(hf);
    return nullptr;
  }
  HANDLE hm = ::CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!hm)
  {
    ::CloseHandle(hf);
    return nullptr;
  }
  void *base = ::MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
  ::CloseHandle(hm); // the view keeps the section object alive
  ::CloseHandle(hf); // the system keeps the file open until the view is unmapped
  if (base)
    *out_len = static_cast<size_t>(sz.QuadPart);
  return base;
#else
  const int fd = ::open(path, O_RDONLY);
  if (fd < 0)
    return nullptr;
  struct stat st;
  if (::fstat(fd, &st) != 0)
  {
    ::close(fd);
    return nullptr;
  }
  const size_t len = static_cast<size_t>(st.st_size);
  void *base = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd); // the mapping holds the file open
  if (base == MAP_FAILED)
    return nullptr;
#if defined(MADV_RANDOM)
  ::madvise(base, len, MADV_RANDOM);
#endif
  *out_len = len;
  return base;
#endif
}

static inline void unmap(void *base, size_t len)
{
  if (!base)
    return;
#if defined(_WIN32)
  (void)len;
  ::UnmapViewOfFile(base);
#else
  ::munmap(base, len);
#endif
}

// A writable file mapping used only while BUILDING the arena (memory-bounded path).
struct rw_map
{
  void *base = nullptr;
  size_t len = 0;
#if defined(_WIN32)
  HANDLE hmap = nullptr;
  HANDLE hfile = nullptr;
#else
  int fd = -1;
#endif
};

// Create `path` (truncating any existing file), size it to `len`, and map it read/write.
// The region is zero-filled. Returns a map whose .base is nullptr on failure.
static inline rw_map map_rw_new(const char *path, size_t len)
{
  rw_map m;
  m.len = len;
#if defined(_WIN32)
  m.hfile = ::CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                          CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (m.hfile == INVALID_HANDLE_VALUE)
  {
    m.hfile = nullptr;
    return m;
  }
  const DWORD len_hi = static_cast<DWORD>(static_cast<uint64_t>(len) >> 32);
  const DWORD len_lo = static_cast<DWORD>(static_cast<uint64_t>(len) & 0xFFFFFFFFull);
  m.hmap = ::CreateFileMappingA(m.hfile, nullptr, PAGE_READWRITE, len_hi, len_lo, nullptr);
  if (!m.hmap)
  {
    ::CloseHandle(m.hfile);
    m.hfile = nullptr;
    return m;
  }
  m.base = ::MapViewOfFile(m.hmap, FILE_MAP_WRITE, 0, 0, 0);
  if (!m.base)
  {
    ::CloseHandle(m.hmap);
    ::CloseHandle(m.hfile);
    m.hmap = m.hfile = nullptr;
  }
  return m;
#else
  m.fd = ::open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (m.fd < 0)
    return m;
  if (::ftruncate(m.fd, static_cast<off_t>(len)) != 0)
  {
    ::close(m.fd);
    m.fd = -1;
    return m;
  }
  m.base = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, m.fd, 0);
  if (m.base == MAP_FAILED)
  {
    m.base = nullptr;
    ::close(m.fd);
    m.fd = -1;
  }
  return m;
#endif
}

// Flush the writable mapping to disk and release it.
static inline void flush_unmap_rw(rw_map &m)
{
  if (!m.base)
    return;
#if defined(_WIN32)
  ::FlushViewOfFile(m.base, 0);
  ::UnmapViewOfFile(m.base);
  if (m.hfile)
    ::FlushFileBuffers(m.hfile);
  if (m.hmap)
    ::CloseHandle(m.hmap);
  if (m.hfile)
    ::CloseHandle(m.hfile);
  m.base = nullptr;
  m.hmap = m.hfile = nullptr;
#else
  ::msync(m.base, m.len, MS_SYNC);
  ::munmap(m.base, m.len);
  if (m.fd >= 0)
  {
    ::fsync(m.fd);
    ::close(m.fd);
  }
  m.base = nullptr;
  m.fd = -1;
#endif
}
} // namespace pds_detail

template <uint8_t N_DIMS, uint16_t N_OPS, typename value_t, typename hash_t>
class point_data_store
{
public:
  using key_t = cell_key_t<N_DIMS>;
  using mapped_type = std::array<value_t, N_OPS>;
  using overlay_t = std::unordered_map<key_t, mapped_type, hash_t>;

  static constexpr size_t NPOS = static_cast<size_t>(-1);
  static constexpr double LOAD_FACTOR = 0.80;
  static constexpr size_t PAGE = 4096;
  static constexpr unsigned char MAGIC[8] = {'D', 'R', 'T', 'S', 'F', 'C', '0', '3'};

  // ---- immutable mmap'd base (open addressing, linear probe) ----
  struct arena_view
  {
    const uint64_t *occ = nullptr; // occupancy bitmap, ceil(C/64) uint64 words
    const int32_t *keys = nullptr; // C * N_DIMS int32, row-major
    const value_t *vals = nullptr; // C * N_OPS value_t, row-major (8-aligned)
    size_t C = 0;                  // capacity (any integer; Lemire-reduced home slot)
    size_t count = 0;              // live entries
    hash_t hasher{};

    bool occupied(size_t s) const { return (occ[s >> 6] >> (s & 63)) & 1ull; }

    key_t key_at(size_t s) const
    {
      key_t k;
      const int32_t *kr = keys + s * static_cast<size_t>(N_DIMS);
      for (uint8_t d = 0; d < N_DIMS; ++d)
        k.idx[d] = kr[d];
      return k;
    }
    const mapped_type &val_at(size_t s) const
    {
      return *reinterpret_cast<const mapped_type *>(vals + s * static_cast<size_t>(N_OPS));
    }
    // Returns slot index of key, or NPOS if absent. Open addressing, linear probe with
    // wraparound (the table always has free slots: C > count), Lemire home placement.
    size_t probe(const key_t &k) const
    {
      if (!C)
        return NPOS;
      size_t s = pds_detail::home_slot(hasher.hash64(k), C);
      for (size_t i = 0; i < C; ++i)
      {
        if (!occupied(s))
          return NPOS; // empty slot terminates the probe (build-once, no tombstones)
        const int32_t *kr = keys + s * static_cast<size_t>(N_DIMS);
        bool eq = true;
        for (uint8_t d = 0; d < N_DIMS; ++d)
          if (kr[d] != k.idx[d]) { eq = false; break; }
        if (eq)
          return s;
        if (++s == C)
          s = 0;
      }
      return NPOS;
    }
  };

  arena_view arena_;
  overlay_t overlay_;

  point_data_store() = default;
  ~point_data_store() { release_mapping(); }
  point_data_store(const point_data_store &) = delete;
  point_data_store &operator=(const point_data_store &) = delete;
  point_data_store(point_data_store &&o) noexcept { move_from(o); }
  point_data_store &operator=(point_data_store &&o) noexcept
  {
    if (this != &o) { release_mapping(); move_from(o); }
    return *this;
  }

private:
  void *map_base_ = nullptr;
  size_t map_len_ = 0;

  void release_mapping()
  {
    if (map_base_)
    {
      pds_detail::unmap(map_base_, map_len_);
      map_base_ = nullptr;
      map_len_ = 0;
    }
  }
  void move_from(point_data_store &o)
  {
    arena_ = o.arena_;
    overlay_ = std::move(o.overlay_);
    map_base_ = o.map_base_;
    map_len_ = o.map_len_;
    o.map_base_ = nullptr;
    o.map_len_ = 0;
    o.arena_ = arena_view{};
  }

  static size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

public:
  // ---- map-like surface used by the interpolators ----
  size_t size() const { return overlay_.size() + arena_.count; }
  void reserve(size_t n) { overlay_.reserve(n > arena_.count ? n - arena_.count : 0); }
  // Wipe overlay AND detach/unmap the arena: restores arena-less overlay-only
  // semantics (used by bulk_set_point_data_arrays = full overlay reload).
  void clear()
  {
    overlay_.clear();
    arena_ = arena_view{};
    release_mapping();
  }
  void detach_arena()
  {
    arena_ = arena_view{};
    release_mapping();
  }
  bool has_arena() const { return arena_.C != 0; }

  // ---- unified const iterator over (overlay ∪ unshadowed arena) ----
  class const_iterator
  {
  public:
    enum stage_t { OVERLAY, ARENA, END };
    const point_data_store *s_ = nullptr;
    stage_t stage_ = END;
    typename overlay_t::const_iterator oit_{};
    size_t aslot_ = 0;

    struct entry
    {
      key_t first;
      const mapped_type &second;
    };
    alignas(entry) mutable unsigned char ebuf_[sizeof(entry)];

    const entry *make(const key_t &k, const mapped_type &v) const
    {
      // entry is trivially destructible -> re-placement-new without destroy is fine.
      return ::new (static_cast<void *>(ebuf_)) entry{k, v};
    }
    const entry *operator->() const
    {
      if (stage_ == OVERLAY)
        return make(oit_->first, oit_->second);
      return make(s_->arena_.key_at(aslot_), s_->arena_.val_at(aslot_));
    }
    const entry &operator*() const { return *operator->(); }

    void skip_to_valid_arena()
    {
      while (aslot_ < s_->arena_.C)
      {
        if (s_->arena_.occupied(aslot_))
        {
          key_t k = s_->arena_.key_at(aslot_);
          if (s_->overlay_.find(k) == s_->overlay_.end())
            return; // occupied AND not shadowed by overlay
        }
        ++aslot_;
      }
      stage_ = END;
    }
    const_iterator &operator++()
    {
      if (stage_ == OVERLAY)
      {
        ++oit_;
        if (oit_ == s_->overlay_.end())
        {
          stage_ = ARENA;
          aslot_ = 0;
          skip_to_valid_arena();
        }
      }
      else if (stage_ == ARENA)
      {
        ++aslot_;
        skip_to_valid_arena();
      }
      return *this;
    }
    bool operator==(const const_iterator &o) const
    {
      if (stage_ != o.stage_)
        return false;
      if (stage_ == OVERLAY)
        return oit_ == o.oit_;
      if (stage_ == ARENA)
        return aslot_ == o.aslot_;
      return true; // both END
    }
    bool operator!=(const const_iterator &o) const { return !(*this == o); }
  };

  const_iterator begin() const
  {
    const_iterator it;
    it.s_ = this;
    if (!overlay_.empty())
    {
      it.stage_ = const_iterator::OVERLAY;
      it.oit_ = overlay_.begin();
    }
    else
    {
      it.stage_ = const_iterator::ARENA;
      it.aslot_ = 0;
      it.skip_to_valid_arena();
    }
    return it;
  }
  const_iterator end() const
  {
    const_iterator it;
    it.s_ = this;
    it.stage_ = const_iterator::END;
    return it;
  }

  const_iterator find(const key_t &k) const
  {
    const_iterator it;
    it.s_ = this;
    auto oit = overlay_.find(k);
    if (oit != overlay_.end())
    {
      it.stage_ = const_iterator::OVERLAY;
      it.oit_ = oit;
      return it;
    }
    const size_t s = arena_.probe(k);
    if (s != NPOS)
    {
      it.stage_ = const_iterator::ARENA;
      it.aslot_ = s;
      return it;
    }
    it.stage_ = const_iterator::END;
    return it;
  }

  // CONCURRENT-READ-SAFE (Phase 2c). Throws std::out_of_range on miss, like map::at.
  const mapped_type &at(const key_t &k) const
  {
    auto oit = overlay_.find(k);
    if (oit != overlay_.end())
      return oit->second;
    const size_t s = arena_.probe(k);
    if (s != NPOS)
      return arena_.val_at(s);
    throw std::out_of_range("point_data_store::at: key not found");
  }

  // OVERLAY-ONLY insert (no arena probe). Callers always precede this with a full
  // find() miss, so an arena-resident key is never duplicated here.
  std::pair<const_iterator, bool> emplace(const key_t &k, const mapped_type &v)
  {
    auto r = overlay_.emplace(k, v);
    const_iterator it;
    it.s_ = this;
    it.stage_ = const_iterator::OVERLAY;
    it.oit_ = r.first;
    return {it, r.second};
  }
  // OVERLAY-ONLY (no arena probe).
  mapped_type &operator[](const key_t &k) { return overlay_[k]; }

  // Wire the arena_view at an externally-owned (mmap'd) region; the store takes
  // ownership of the mapping (munmap on destruct) when map_base != nullptr.
  void attach_arena(const uint64_t *occ, const int32_t *keys, const value_t *vals,
                    size_t C, size_t count, void *map_base = nullptr, size_t map_len = 0)
  {
    // Re-pointing the arena: always drop any previously-owned mapping first (callers never
    // alias the new arena into the old mapping), then take ownership of the new one. A null
    // map_base means the arena points at caller-owned memory (e.g. a test buffer) -> no
    // ownership taken.
    release_mapping();
    arena_.occ = occ;
    arena_.keys = keys;
    arena_.vals = vals;
    arena_.C = C;
    arena_.count = count;
    map_base_ = map_base;
    map_len_ = map_len;
  }

  // ---- arena fingerprint: changes if the hash64 output, the slot-PLACEMENT algorithm,
  // N_DIMS or N_OPS change. An arena whose recorded hash_id != this is never mis-probed
  // (the loader rebuilds it from a placement-independent occupied-slot scan). PLACEMENT_VER
  // bumps whenever the home/probe scheme changes (v2 = Lemire reduction, non-power-of-two).
  static constexpr uint64_t PLACEMENT_VER = 2;
  static uint64_t arena_hash_id()
  {
    hash_t h{};
    uint64_t f = 0x4f424c4641524e41ULL; // "OBLFARNA"
    const int32_t golden[5] = {0, 1, -1, 2147483647, -2147483647 - 1};
    for (int g = 0; g < 5; ++g)
    {
      key_t k;
      for (uint8_t d = 0; d < N_DIMS; ++d)
        k.idx[d] = golden[(g + d) % 5];
      f = f * 1000003ull ^ h.hash64(k);
    }
    f ^= (static_cast<uint64_t>(N_DIMS) << 40) ^ (static_cast<uint64_t>(N_OPS) << 8);
    f = f * 1000003ull ^ (PLACEMENT_VER * 0x9e3779b97f4a7c15ULL);
    return f;
  }

  // ---- build the cache file (arena) from the live union (compaction/migration) ----
  // Writes a COMPLETE arena file to `path` (caller renames atomically). vals crc is
  // omitted (0): the file is written to a temp path + os.replace, so the base is
  // never torn in place. Cross-platform (POSIX mmap / Win32 file mapping).
  struct arena_layout_t
  {
    size_t C, count, header_len;
    size_t bitmap_off, bitmap_len, keys_off, keys_len, vals_off, vals_len, arena_end;
    char json[1024];
  };

  // Compute the arena layout + JSON header for the current live count.
  arena_layout_t _arena_layout(uint64_t hash_id) const
  {
    arena_layout_t L;
    L.count = 0;
    for (auto it = begin(); it != end(); ++it)
      ++L.count;
    // Tight capacity at LOAD_FACTOR (Lemire reduction allows non-power-of-two C). The
    // max(count+1, ...) guarantees C > count so the linear probe always finds a free slot.
    if (L.count == 0)
      L.C = 1;
    else
    {
      const size_t lf_cap = static_cast<size_t>(L.count / LOAD_FACTOR) + 1;
      L.C = (L.count + 1 > lf_cap) ? (L.count + 1) : lf_cap;
    }
    L.bitmap_off = PAGE;
    L.bitmap_len = ((L.C + 63) / 64) * 8;
    L.keys_off = align_up(L.bitmap_off + L.bitmap_len, PAGE);
    L.keys_len = L.C * static_cast<size_t>(N_DIMS) * 4;
    L.vals_off = align_up(L.keys_off + L.keys_len, PAGE);
    L.vals_len = L.C * static_cast<size_t>(N_OPS) * sizeof(value_t);
    L.arena_end = align_up(L.vals_off + L.vals_len, PAGE);
    int hl = std::snprintf(
        L.json, sizeof(L.json),
        "{\"format\":3,\"n_dims\":%d,\"n_ops\":%d,\"val_dtype\":\"<f8\","
        "\"key_dtype\":\"<i4\",\"epoch_dtype\":\"<u8\",\"endian\":\"L\","
        "\"hash_id\":%llu,\"arena_capacity\":%llu,\"arena_count\":%llu,"
        "\"page_size\":%llu,\"bitmap_off\":%llu,\"bitmap_len\":%llu,"
        "\"keys_off\":%llu,\"keys_len\":%llu,\"vals_off\":%llu,\"vals_len\":%llu,"
        "\"arena_end\":%llu,\"arena_crc32\":0}",
        static_cast<int>(N_DIMS), static_cast<int>(N_OPS),
        (unsigned long long)hash_id, (unsigned long long)L.C, (unsigned long long)L.count,
        (unsigned long long)PAGE, (unsigned long long)L.bitmap_off, (unsigned long long)L.bitmap_len,
        (unsigned long long)L.keys_off, (unsigned long long)L.keys_len,
        (unsigned long long)L.vals_off, (unsigned long long)L.vals_len,
        (unsigned long long)L.arena_end);
    // snprintf returns the length it WOULD have written: guard both buffer truncation
    // (hl >= sizeof(json), else the header is silently cut) and overrun of the first page.
    if (hl < 0 || static_cast<size_t>(hl) >= sizeof(L.json) ||
        static_cast<size_t>(16 + hl) > L.bitmap_off)
      throw std::runtime_error("point_data_store::build_arena_file: header too large");
    L.header_len = static_cast<size_t>(hl);
    return L;
  }

  // Pass 2: place every live entry by hash (linear probe to first clear bit) into the
  // (RAM or mmap) arena buffers. Shared by both build paths. MUST match arena_view::probe
  // (Lemire home + linear-probe wraparound) so the on-disk placement is readable.
  void _place_into(uint64_t *occ, int32_t *okeys, value_t *ovals, size_t C) const
  {
    hash_t hasher{};
    for (auto it = begin(); it != end(); ++it)
    {
      const entry_ref e = deref(it);
      size_t s = pds_detail::home_slot(hasher.hash64(e.key), C);
      while (occ[s >> 6] & (1ull << (s & 63)))
        if (++s == C)
          s = 0;
      occ[s >> 6] |= (1ull << (s & 63));
      int32_t *kr = okeys + s * static_cast<size_t>(N_DIMS);
      for (uint8_t d = 0; d < N_DIMS; ++d)
        kr[d] = e.key.idx[d];
      value_t *vr = ovals + s * static_cast<size_t>(N_OPS);
      for (uint16_t op = 0; op < N_OPS; ++op)
        vr[op] = e.val[op];
    }
  }

  static bool _arena_fits_in_ram(size_t bytes)
  {
    const size_t avail = pds_detail::available_ram_bytes();
    if (avail == 0)
      return false; // unknown available RAM -> use the memory-bounded mmap builder
    return static_cast<long double>(bytes) * 1.2L < static_cast<long double>(avail); // 20% headroom
  }

  static void _write_all(std::FILE *f, const void *buf, size_t n)
  {
    const char *p = static_cast<const char *>(buf);
    while (n)
    {
      const size_t chunk = n > (size_t(1) << 30) ? (size_t(1) << 30) : n; // <=1 GiB/write
      if (std::fwrite(p, 1, chunk, f) != chunk)
        throw std::runtime_error("point_data_store: write failed");
      p += chunk;
      n -= chunk;
    }
  }

  static void _write_zeros(std::FILE *f, size_t n)
  {
    static const char z[PAGE] = {0};
    while (n)
    {
      const size_t c = n < sizeof(z) ? n : sizeof(z);
      _write_all(f, z, c);
      n -= c;
    }
  }

  // FAST builder: place the arena into anonymous RAM (random writes hit RAM, not random
  // disk page-faults), then stream it to disk in one big SEQUENTIAL write. Returns false
  // if the RAM buffers cannot be allocated (caller falls back to the mmap builder).
  // calloc keeps unoccupied slots as zero pages (committed lazily), so peak RAM tracks the
  // ~LOAD_FACTOR-occupied fraction, and the on-disk bytes are identical to the mmap path.
  bool _build_arena_ram(const std::string &path, const arena_layout_t &L) const
  {
    struct free_guard
    {
      void *p = nullptr;
      ~free_guard() { if (p) ::free(p); }
    };
    free_guard occ_g, keys_g, vals_g;
    occ_g.p = ::calloc(L.bitmap_len / 8, sizeof(uint64_t));
    keys_g.p = ::calloc(L.C * static_cast<size_t>(N_DIMS), sizeof(int32_t));
    vals_g.p = ::calloc(L.C * static_cast<size_t>(N_OPS), sizeof(value_t));
    if (!occ_g.p || !keys_g.p || !vals_g.p)
      return false; // not enough RAM -> mmap fallback

    _place_into(static_cast<uint64_t *>(occ_g.p), static_cast<int32_t *>(keys_g.p),
                static_cast<value_t *>(vals_g.p), L.C);

    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f)
      throw std::runtime_error("point_data_store::_build_arena_ram: open failed");
    try
    {
      _write_all(f, MAGIC, 8);
      const uint64_t hl = L.header_len;
      _write_all(f, &hl, 8);
      _write_all(f, L.json, L.header_len);
      _write_zeros(f, L.bitmap_off - (16 + L.header_len));
      _write_all(f, occ_g.p, L.bitmap_len);
      _write_zeros(f, L.keys_off - (L.bitmap_off + L.bitmap_len));
      _write_all(f, keys_g.p, L.keys_len);
      _write_zeros(f, L.vals_off - (L.keys_off + L.keys_len));
      _write_all(f, vals_g.p, L.vals_len);
      _write_zeros(f, L.arena_end - (L.vals_off + L.vals_len));
      pds_detail::fsync_file(f);
    }
    catch (...)
    {
      std::fclose(f);
      throw;
    }
    std::fclose(f);
    return true;
  }

  // MEMORY-BOUNDED builder: place directly into the mmap'd output file. Random page-faults
  // make this slower than the RAM builder, but its footprint is reclaimable page cache, so
  // it is the safe fallback when the arena does not comfortably fit in RAM.
  void _build_arena_mmap(const std::string &path, const arena_layout_t &L) const
  {
    pds_detail::rw_map m = pds_detail::map_rw_new(path.c_str(), L.arena_end);
    if (!m.base)
      throw std::runtime_error("point_data_store::_build_arena_mmap: map failed");
    char *b = static_cast<char *>(m.base);
    std::memcpy(b, MAGIC, 8);
    const uint64_t hlen64 = L.header_len;
    std::memcpy(b + 8, &hlen64, 8);
    std::memcpy(b + 16, L.json, L.header_len); // the mapping is zero-filled, so the rest stays 0
    _place_into(reinterpret_cast<uint64_t *>(b + L.bitmap_off),
                reinterpret_cast<int32_t *>(b + L.keys_off),
                reinterpret_cast<value_t *>(b + L.vals_off), L.C);
    pds_detail::flush_unmap_rw(m);
  }

  // Build a COMPLETE arena file at `path` (caller renames atomically). Prefers the fast
  // RAM-buffer builder when the arena fits comfortably in available RAM, else falls back
  // to the memory-bounded mmap builder. Both produce byte-identical files. Set the env var
  // OBL_CACHE_BUILD_MMAP to force the mmap builder.
  void build_arena_file(const std::string &path, uint64_t hash_id) const
  {
    const arena_layout_t L = _arena_layout(hash_id);
    const bool force_mmap = std::getenv("OBL_CACHE_BUILD_MMAP") != nullptr;
    if (!force_mmap && _arena_fits_in_ram(L.arena_end))
    {
      if (_build_arena_ram(path, L))
        return;
    }
    _build_arena_mmap(path, L);
  }

  // mmap an arena written by build_arena_file (offsets parsed by Python and
  // passed in). Read-only, private mapping; the store owns it (munmap on destruct).
  void mmap_arena_at(const std::string &path, size_t bitmap_off, size_t keys_off,
                     size_t vals_off, size_t C, size_t count)
  {
    size_t len = 0;
    void *base = pds_detail::map_readonly(path.c_str(), &len);
    if (!base)
      throw std::runtime_error("point_data_store::mmap_arena_at: map failed");
    char *b = static_cast<char *>(base);
    attach_arena(reinterpret_cast<const uint64_t *>(b + bitmap_off),
                 reinterpret_cast<const int32_t *>(b + keys_off),
                 reinterpret_cast<const value_t *>(b + vals_off),
                 C, count, base, len);
  }

private:
  struct entry_ref
  {
    key_t key;
    const mapped_type &val;
  };
  // Single-deref helper so build_arena_file never calls operator-> twice (which
  // would rebuild the iterator's proxy buffer mid-expression).
  entry_ref deref(const const_iterator &it) const
  {
    const auto &e = *it;
    return entry_ref{e.first, e.second};
  }
};

#endif // DARTS_POINT_DATA_STORE_HPP
