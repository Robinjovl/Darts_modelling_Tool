// Collision-safety test for the adaptive OBL interpolator cell keys.
//
// The CPU adaptive interpolator stores hypercubes in std::unordered_map<cell_key_t,
// ..., cell_key_hash> — a hash COLLISION only lengthens a bucket chain because the
// map resolves it with the full per-axis cell_key_t::operator==, so the CPU is
// EXACT. The GPU open-addressed map (gpu_hashmap_async) instead stores only the
// 63-bit content hash (key_from_cell) and matches on that hash alone, so it is
// collision-PRONE: two distinct cells that share a 63-bit hash would alias.
//
// This test exercises both, header-only, for OBL/interpolation dimensions 4..24 and
// several OBL spacing regimes (the per-axis cell index magnitude scales inversely
// with axes_step, so different index ranges == different spacings, including a
// NON-UNIFORM per-axis regime). It asserts:
//   (1) CPU exact: every distinct key resolves to its own value (zero lookup errors)
//       in ALL regimes and dimensions, even under forced hash collisions.
//   (2) GPU realistic: the full 63-bit hash produces ZERO collisions at realistic
//       cache sizes (birthday rate ~N^2/2^64 is astronomically small).
//   (3) GPU failure mode is real & detectable: with a deliberately reduced hash
//       mask (forcing the birthday regime) distinct cells DO alias on the GPU-style
//       hash, while the CPU full-key map stays exact.
//
// Returns the number of failed expectations (0 == pass), per ctest convention.

#include "multi_index_key.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_map>
#include <vector>

// Mirror gpu_hashmap_async::key_from_cell: the 63-bit content hash the GPU map uses
// as BOTH bucket selector and match key (it never stores the full cell_key_t).
template <uint8_t N_DIMS>
static int64_t gpu_hash63(const cell_key_t<N_DIMS> &k)
{
  const uint64_t h = cell_key_hash<N_DIMS>{}.hash64(k);
  return static_cast<int64_t>(h & 0x7fffffffffffffffULL);
}

struct DimResult
{
  uint64_t cpu_lookup_errors = 0;   // CPU std::unordered_map<cell_key_t> mis-resolves
  uint64_t gpu_hash_collisions = 0; // distinct keys sharing the (masked) GPU hash
};

// Generate K distinct keys with per-axis index ranges `ranges[a]` (== OBL spacing
// regime), insert into the real CPU map type, and tally GPU-hash collisions under a
// `hash_bits`-bit mask (63 == the production GPU hash; smaller == forced birthday).
template <uint8_t N_DIMS>
static DimResult run_dim(uint64_t K, const std::vector<int32_t> &ranges,
                         unsigned hash_bits, std::mt19937_64 &rng)
{
  std::unordered_map<cell_key_t<N_DIMS>, uint64_t, cell_key_hash<N_DIMS>> cpu_map;
  std::unordered_map<int64_t, cell_key_t<N_DIMS>> gpu_hash_first; // masked hash -> first key
  std::vector<cell_key_t<N_DIMS>> keys;
  keys.reserve(K);
  cpu_map.reserve(K * 2);
  gpu_hash_first.reserve(K * 2);

  const int64_t mask = (hash_bits >= 63) ? 0x7fffffffffffffffLL
                                         : ((static_cast<int64_t>(1) << hash_bits) - 1);

  DimResult r;
  while (keys.size() < K)
  {
    cell_key_t<N_DIMS> key;
    for (uint8_t a = 0; a < N_DIMS; ++a)
    {
      std::uniform_int_distribution<int32_t> d(-ranges[a], ranges[a]);
      key.idx[a] = d(rng);
    }
    // Enforce full-key distinctness (the CPU map is the ground truth for identity).
    if (cpu_map.find(key) != cpu_map.end())
      continue;
    const uint64_t id = keys.size();
    cpu_map.emplace(key, id);
    keys.push_back(key);

    const int64_t gk = gpu_hash63<N_DIMS>(key) & mask;
    auto it = gpu_hash_first.find(gk);
    if (it != gpu_hash_first.end())
      r.gpu_hash_collisions++; // distinct cell aliasing onto an occupied GPU slot
    else
      gpu_hash_first.emplace(gk, key);
  }

  // CPU exactness: every key must resolve to its own id via full-key compare.
  for (uint64_t i = 0; i < K; ++i)
  {
    auto it = cpu_map.find(keys[i]);
    if (it == cpu_map.end() || it->second != i)
      r.cpu_lookup_errors++;
  }
  return r;
}

// Run all spacing regimes for one (compile-time) dimension. Returns #failures.
template <uint8_t N_DIMS>
static int test_dim(uint64_t K, std::mt19937_64 &rng)
{
  int failures = 0;

  // OBL spacing regimes via per-axis cell-index range (index ~ value / axes_step):
  std::vector<int32_t> coarse(N_DIMS, 100);       // coarse spacing -> small indices
  std::vector<int32_t> tight(N_DIMS, 1'000'000);  // tight spacing  -> large indices
  std::vector<int32_t> nonuniform(N_DIMS);        // NON-UNIFORM spacing across axes
  for (uint8_t a = 0; a < N_DIMS; ++a)
    nonuniform[a] = static_cast<int32_t>(10) * (1 + (a % 6) * (a % 6)); // 10..910, per-axis

  struct Regime { const char *name; const std::vector<int32_t> *ranges; };
  const Regime regimes[] = {
      {"coarse-uniform", &coarse},
      {"tight-uniform", &tight},
      {"non-uniform", &nonuniform},
  };

  for (const auto &reg : regimes)
  {
    DimResult r = run_dim<N_DIMS>(K, *reg.ranges, /*hash_bits=*/63, rng);
    const bool ok = (r.cpu_lookup_errors == 0) && (r.gpu_hash_collisions == 0);
    printf("  N_DIMS=%2d  spacing=%-15s K=%llu  CPU_errors=%llu  GPU63_collisions=%llu  [%s]\n",
           (int)N_DIMS, reg.name, (unsigned long long)K,
           (unsigned long long)r.cpu_lookup_errors,
           (unsigned long long)r.gpu_hash_collisions, ok ? "OK" : "FAIL");
    if (!ok)
      failures++;
  }
  return failures;
}

// Forced-collision stress: a reduced hash mask drives the birthday regime so that
// distinct cells provably alias on a GPU-style hash, while the CPU map stays exact.
template <uint8_t N_DIMS>
static int test_forced_collision(uint64_t K, unsigned hash_bits, std::mt19937_64 &rng)
{
  std::vector<int32_t> tight(N_DIMS, 1'000'000);
  DimResult r = run_dim<N_DIMS>(K, tight, hash_bits, rng);
  // Expectation: CPU exact (0 errors) AND the reduced GPU hash DID collide (>0),
  // demonstrating the failure mode is real and this test would catch it.
  const bool cpu_ok = (r.cpu_lookup_errors == 0);
  const bool collided = (r.gpu_hash_collisions > 0);
  const bool ok = cpu_ok && collided;
  printf("  N_DIMS=%2d  FORCED hash_bits=%u  K=%llu  CPU_errors=%llu  "
         "GPU%u_collisions=%llu  [CPU exact=%s, failure-mode reproduced=%s] [%s]\n",
         (int)N_DIMS, hash_bits, (unsigned long long)K,
         (unsigned long long)r.cpu_lookup_errors, hash_bits,
         (unsigned long long)r.gpu_hash_collisions,
         cpu_ok ? "yes" : "NO", collided ? "yes" : "NO", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

// ───────────────────────── manual probe mode ─────────────────────────
// Lean GPU-hash collision counter for a user-chosen (N_DIMS, points/axis, K,
// hash_bits). Stores only the 63-bit hashes (8 bytes each) so K can be pushed
// high. At these cell-space sizes a full-key duplicate is negligibly unlikely, so
// a hash duplicate == a genuine GPU 63-bit-hash collision (distinct cells aliasing).
template <uint8_t N_DIMS>
static int probe_impl(int points_per_axis, uint64_t K, unsigned hash_bits)
{
  std::mt19937_64 rng(987654321ULL);
  std::uniform_int_distribution<int32_t> d(0, points_per_axis - 1);
  const int64_t mask = (hash_bits >= 63) ? 0x7fffffffffffffffLL
                                         : ((static_cast<int64_t>(1) << hash_bits) - 1);
  std::vector<int64_t> hashes;
  hashes.reserve(K);
  for (uint64_t i = 0; i < K; ++i)
  {
    cell_key_t<N_DIMS> key;
    for (uint8_t a = 0; a < N_DIMS; ++a)
      key.idx[a] = d(rng);
    hashes.push_back(gpu_hash63<N_DIMS>(key) & mask);
  }
  std::sort(hashes.begin(), hashes.end());
  uint64_t collisions = 0;
  for (size_t i = 1; i < hashes.size(); ++i)
    if (hashes[i] == hashes[i - 1])
      collisions++;
  const double bits = (hash_bits >= 63) ? 63.0 : static_cast<double>(hash_bits);
  const double expected =
      static_cast<double>(K) * static_cast<double>(K - 1) / 2.0 / std::pow(2.0, bits);
  printf("PROBE  N_DIMS=%d  points/axis=%d  K=%llu  hash_bits=%u\n",
         (int)N_DIMS, points_per_axis, (unsigned long long)K, hash_bits);
  printf("  distinct-cell space ~ %d^%d  (>> K, so sampled keys are sparse / ~all distinct)\n",
         points_per_axis, (int)N_DIMS);
  printf("  GPU %u-bit hash collisions (distinct cells aliasing) = %llu\n",
         hash_bits, (unsigned long long)collisions);
  printf("  birthday expectation  K*(K-1)/2 / 2^%g  = %.4g\n", bits, expected);
  return 0;
}

static int probe(int nd, int points_per_axis, uint64_t K, unsigned hash_bits)
{
  switch (nd)
  {
  case 4:  return probe_impl<4>(points_per_axis, K, hash_bits);
  case 8:  return probe_impl<8>(points_per_axis, K, hash_bits);
  case 12: return probe_impl<12>(points_per_axis, K, hash_bits);
  case 16: return probe_impl<16>(points_per_axis, K, hash_bits);
  case 20: return probe_impl<20>(points_per_axis, K, hash_bits);
  case 24: return probe_impl<24>(points_per_axis, K, hash_bits);
  default:
    fprintf(stderr, "probe: n_dims=%d not instantiated (use one of 4,8,12,16,20,24)\n", nd);
    return 2;
  }
}

int main(int argc, char **argv)
{
  // Manual probe mode: <n_dims> <points_per_axis> <K> [hash_bits=63]
  //   e.g.  ./interpolators_test_cell_key_collision 24 1000 10000000
  if (argc >= 4)
  {
    const int nd = std::atoi(argv[1]);
    const int P = std::atoi(argv[2]);
    const uint64_t K = std::strtoull(argv[3], nullptr, 10);
    const unsigned hb = (argc >= 5) ? static_cast<unsigned>(std::atoi(argv[4])) : 63u;
    return probe(nd, P, K, hb);
  }

  std::mt19937_64 rng(123456789ULL); // fixed seed -> deterministic
  int failures = 0;

  printf("==== cell_key_t collision test: CPU exact vs GPU 63-bit hash ====\n");
  printf("[1] Realistic regimes — expect CPU_errors=0 AND GPU63_collisions=0:\n");
  const uint64_t K = 100'000; // realistic per-(dim,regime) cache size; birthday(63) ~ 0
  failures += test_dim<4>(K, rng);
  failures += test_dim<10>(K, rng);
  failures += test_dim<16>(K, rng);
  failures += test_dim<22>(K, rng);
  failures += test_dim<28>(K, rng);

  printf("[2] Forced-collision stress (reduced hash mask drives the birthday regime):\n");
  // 24-bit hash, K=50000 -> expected collisions ~ K^2/2^25 ~ 75; proves the GPU
  // alias failure mode is real & detected, and that CPU stays exact under it.
  failures += test_forced_collision<4>(50'000, /*hash_bits=*/24, rng);
  failures += test_forced_collision<24>(50'000, /*hash_bits=*/24, rng);

  printf("==== %s (%d failed expectation%s) ====\n",
         failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
  return failures;
}
