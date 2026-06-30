// Standalone unit test for point_data_store (Phase F / FC03 hybrid arena+overlay).
// Build:  g++ -std=c++20 -O2 -I interpolation/include tests/interpolators/cpp/test_point_data_store.cpp -o /tmp/t && /tmp/t
#undef NDEBUG  // keep assert() active even in a Release/NDEBUG ctest build
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "point_data_store.hpp"

static constexpr uint8_t ND = 3;
static constexpr uint16_t NO = 4;
using store_t = point_data_store<ND, NO, double, cell_key_hash<ND>>;
using K = cell_key_t<ND>;
using V = std::array<double, NO>;

static K mk(int a, int b, int c)
{
  K k;
  k.idx = {a, b, c};
  return k;
}
static V mv(double base)
{
  V v;
  for (uint16_t i = 0; i < NO; ++i)
    v[i] = base + 0.25 * i;
  return v;
}
struct kcmp
{
  bool operator()(const K &a, const K &b) const { return a.idx < b.idx; }
};

// minimal JSON int field reader for the FC03 header
static uint64_t jint(const std::string &s, const char *field)
{
  std::string pat = std::string("\"") + field + "\":";
  size_t p = s.find(pat);
  assert(p != std::string::npos);
  return std::strtoull(s.c_str() + p + pat.size(), nullptr, 10);
}

static void read_header(const std::string &path, uint64_t &bo, uint64_t &ko, uint64_t &vo,
                        uint64_t &C, uint64_t &cnt, uint64_t &hash_id)
{
  FILE *f = std::fopen(path.c_str(), "rb");
  assert(f);
  char magic[8];
  assert(std::fread(magic, 1, 8, f) == 8);
  assert(std::memcmp(magic, "DRTSFC03", 8) == 0);
  uint64_t hlen;
  assert(std::fread(&hlen, 1, 8, f) == 8);
  std::string json(hlen, '\0');
  assert(std::fread(json.data(), 1, hlen, f) == hlen);
  std::fclose(f);
  bo = jint(json, "bitmap_off");
  ko = jint(json, "keys_off");
  vo = jint(json, "vals_off");
  C = jint(json, "arena_capacity");
  cnt = jint(json, "arena_count");
  hash_id = jint(json, "hash_id");
}

// Golden cell_key_hash::hash64 vectors. hash64 is the FC03 on-disk slot-placement
// function: if these change, every existing FC03 arena's placement is invalid. The
// arena_hash_id() fingerprint already auto-detects this (old arenas are safely
// rebuilt, never mis-probed), but this pin makes an *accidental* hash change a loud
// CI failure so the version tag is bumped deliberately.
static void test_golden_hash()
{
  cell_key_hash<3> h3;
  cell_key_hash<8> h8;
  auto k3 = [&](int a, int b, int c) { cell_key_t<3> k; k.idx = {a, b, c}; return h3.hash64(k); };
  assert(k3(0, 0, 0) == 7211264085254537017ULL);
  assert(k3(1, 2, 3) == 1052850491782272769ULL);
  assert(k3(-1, -2, -3) == 10043956612046299227ULL);
  cell_key_t<8> k8;
  k8.idx = {1, -2, 3, -4, 5, -6, 7, -8};
  assert(h8.hash64(k8) == 7822504572828504668ULL);
}

int main()
{
  test_golden_hash();
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> kd(-200, 200);

  // reference union (key -> val) and a "base" subset to bake into the arena.
  std::map<K, V, kcmp> ref;
  store_t s1;
  const int N_BASE = 5000;
  for (int i = 0; i < N_BASE; ++i)
  {
    K k = mk(kd(rng), kd(rng), kd(rng));
    V v = mv(i + 1);
    if (ref.emplace(k, v).second)
      s1.emplace(k, v); // overlay-only; keep ref & overlay in sync on duplicate draws
  }
  assert(s1.size() == s1.overlay_.size());
  const uint64_t hid = store_t::arena_hash_id();

  // build arena file from the overlay (migration: overlay -> arena)
  std::string path = "/tmp/_fc3_test_arena.bin";
  s1.build_arena_file(path, hid);

  // The fast RAM-buffer builder and the memory-bounded mmap builder must produce
  // BYTE-IDENTICAL files (same placement, same layout, zeros in unoccupied slots).
  {
    std::string ram_path = "/tmp/_fc3_test_ram.bin";
    std::string mmap_path = "/tmp/_fc3_test_mmap.bin";
    ::unsetenv("OBL_FC3_BUILD_MMAP");
    s1.build_arena_file(ram_path, hid); // RAM path (small arena -> fits)
    ::setenv("OBL_FC3_BUILD_MMAP", "1", 1);
    s1.build_arena_file(mmap_path, hid); // forced mmap path
    ::unsetenv("OBL_FC3_BUILD_MMAP");
    auto slurp = [](const std::string &p) {
      FILE *f = std::fopen(p.c_str(), "rb");
      assert(f);
      std::fseek(f, 0, SEEK_END);
      long n = std::ftell(f);
      std::fseek(f, 0, SEEK_SET);
      std::string buf(static_cast<size_t>(n), '\0');
      assert(std::fread(buf.data(), 1, static_cast<size_t>(n), f) == static_cast<size_t>(n));
      std::fclose(f);
      return buf;
    };
    assert(slurp(ram_path) == slurp(mmap_path)); // RAM and mmap builders agree byte-for-byte
    std::remove(ram_path.c_str());
    std::remove(mmap_path.c_str());
    std::printf("OK build parity: RAM-buffer builder == mmap builder (byte-identical)\n");
  }

  // load into a fresh store via mmap
  uint64_t bo, ko, vo, C, cnt, file_hid;
  read_header(path, bo, ko, vo, C, cnt, file_hid);
  assert(file_hid == hid);
  assert(cnt == ref.size()); // exact deduped count
  store_t s2;
  s2.mmap_arena_at(path, bo, ko, vo, C, cnt);
  assert(s2.has_arena());
  assert(s2.overlay_.empty());
  assert(s2.size() == ref.size());

  // every base key found in arena with correct value; at() and find() agree
  for (auto &kv : ref)
  {
    auto it = s2.find(kv.first);
    assert(it != s2.end());
    const auto &got = s2.at(kv.first);
    for (uint16_t op = 0; op < NO; ++op)
      assert(got[op] == kv.second[op]);
    assert(it->second[0] == kv.second[0]);
  }
  // a miss
  bool threw = false;
  try { (void)s2.at(mk(999999, 1, 1)); } catch (const std::out_of_range &) { threw = true; }
  assert(threw);
  assert(s2.find(mk(999999, 1, 1)) == s2.end());

  // add deltas into the overlay (new points this run, disjoint from base)
  std::map<K, V, kcmp> deltas;
  for (int i = 0; i < 1500; ++i)
  {
    K k = mk(1000 + i, 2000 + i, -3000 - i); // disjoint range
    V v = mv(100000 + i);
    s2.emplace(k, v);
    deltas[k] = v;
    ref[k] = v;
  }
  assert(s2.size() == ref.size()); // overlay disjoint from arena -> exact

  // lookups across both layers
  for (auto &kv : ref)
  {
    const auto &got = s2.at(kv.first);
    for (uint16_t op = 0; op < NO; ++op)
      assert(got[op] == kv.second[op]);
  }

  // iterate the union -> must equal ref exactly (dedup, no dup, no miss)
  std::map<K, V, kcmp> seen;
  for (auto it = s2.begin(); it != s2.end(); ++it)
  {
    K k = it->first;
    assert(seen.find(k) == seen.end()); // no duplicates
    V v;
    for (uint16_t op = 0; op < NO; ++op)
      v[op] = it->second[op];
    seen[k] = v;
  }
  assert(seen.size() == ref.size());
  for (auto &kv : ref)
  {
    assert(seen.count(kv.first));
    for (uint16_t op = 0; op < NO; ++op)
      assert(seen[kv.first][op] == kv.second[op]);
  }

  // overlay shadows arena: overwrite a base key via operator[] -> overlay wins, no dup
  K shadow = ref.begin()->first;
  V newv = mv(424242);
  s2[shadow] = newv;
  const auto &g = s2.at(shadow);
  for (uint16_t op = 0; op < NO; ++op)
    assert(g[op] == newv[op]);
  size_t union_cnt = 0;
  for (auto it = s2.begin(); it != s2.end(); ++it)
    ++union_cnt;
  assert(union_cnt == ref.size()); // shadowed arena slot not double-counted in iteration

  // compaction: rebuild arena from (arena ∪ overlay), reload, verify union preserved
  std::string path2 = "/tmp/_fc3_test_arena2.bin";
  ref[shadow] = newv; // reflect the shadow overwrite in the reference
  s2.build_arena_file(path2, hid);
  uint64_t bo2, ko2, vo2, C2, cnt2, hid2;
  read_header(path2, bo2, ko2, vo2, C2, cnt2, hid2);
  assert(cnt2 == ref.size());
  store_t s3;
  s3.mmap_arena_at(path2, bo2, ko2, vo2, C2, cnt2);
  assert(s3.size() == ref.size());
  for (auto &kv : ref)
  {
    const auto &got = s3.at(kv.first);
    for (uint16_t op = 0; op < NO; ++op)
      assert(got[op] == kv.second[op]);
  }

  // clear() detaches arena + wipes overlay
  s3.clear();
  assert(!s3.has_arena());
  assert(s3.size() == 0);

  std::remove(path.c_str());
  std::remove(path2.c_str());
  std::printf("OK point_data_store: base=%zu deltas=%zu union=%zu (build->mmap->lookup->iterate->compact->clear)\n",
              (size_t)N_BASE, deltas.size(), ref.size());
  return 0;
}
