#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>


static std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

class Hash64 {
public:
    explicit Hash64(std::uint64_t seed) : seed_(seed) {}
    std::uint64_t operator()(std::string_view s) const {
        std::uint64_t h = 1469598103934665603ull ^ seed_;
        for (unsigned char c : s) {
            h ^= std::uint64_t(c);
            h *= 1099511628211ull;
        }
        return splitmix64(h);
    }
private:
    std::uint64_t seed_;
};

class RandomStreamGenerator {
public:
    RandomStreamGenerator(std::uint64_t seed, int maxLen)
        : rng_(seed), maxLen_(maxLen) {
        alphabet_ =
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789";
    }

    std::vector<std::string> generate(std::size_t n) {
        std::uniform_int_distribution<int> lenDist(1, maxLen_);
        std::uniform_int_distribution<int> chDist(0, int(alphabet_.size()) - 1);

        std::vector<std::string> stream;
        stream.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            int len = lenDist(rng_);
            std::string s;
            s.resize(len);
            for (int j = 0; j < len; ++j) s[j] = alphabet_[chDist(rng_)];
            stream.push_back(std::move(s));
        }
        return stream;
    }

    static std::vector<std::size_t> checkpoints_by_percent(std::size_t n, double stepPercent) {
        std::vector<std::size_t> cps;
        if (n == 0) return cps;
        if (stepPercent <= 0) stepPercent = 5.0;

        for (double p = stepPercent; p <= 100.0 + 1e-12; p += stepPercent) {
            std::size_t t = std::size_t(std::llround((p / 100.0) * double(n)));
            t = std::max<std::size_t>(1, std::min<std::size_t>(t, n));
            if (cps.empty() || cps.back() != t) cps.push_back(t);
        }
        if (cps.empty() || cps.back() != n) cps.push_back(n);
        cps.erase(std::unique(cps.begin(), cps.end()), cps.end());
        std::sort(cps.begin(), cps.end());
        return cps;
    }

private:
    std::mt19937_64 rng_;
    int maxLen_;
    std::string alphabet_;
};

static double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
}

static double stdev_sample(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double mu = mean(v);
    double acc = 0.0;
    for (double x : v) {
        double d = x - mu;
        acc += d * d;
    }
    return std::sqrt(acc / double(v.size() - 1));
}

static void ensure_dir(const std::string& path) {
    std::filesystem::create_directories(path);
}

struct Point {
    std::size_t t = 0;
    std::size_t trueF0 = 0;
    double est = 0.0;
};

struct StatsRow {
    std::size_t step = 0;
    std::size_t t = 0;
    double fraction = 0.0;
    double mean_true = 0.0;
    double mean_est = 0.0;
    double std_est = 0.0;
    double rel_bias = 0.0;
    double rel_std = 0.0;
    double th_1_04 = 0.0;
    double th_1_3 = 0.0;
};

struct AnalysisSummary {
    double rel_std_mean = 0.0;
    double rel_std_max = 0.0;
    double rel_bias_last = 0.0;
    double rel_bias_max_abs = 0.0;
    double th104 = 0.0;
    double th13 = 0.0;
};

static std::vector<StatsRow> build_stats(
    const std::vector<std::vector<Point>>& runs,
    const std::vector<std::size_t>& checkpoints,
    std::size_t streamLen,
    int B
) {
    std::vector<StatsRow> stats;
    stats.reserve(checkpoints.size());

    double m = double(1ull << B);
    double th104 = 1.04 / std::sqrt(m);
    double th13  = 1.3  / std::sqrt(m);

    for (std::size_t idx = 0; idx < checkpoints.size(); ++idx) {
        std::vector<double> ests, trues;
        ests.reserve(runs.size());
        trues.reserve(runs.size());

        for (const auto& run : runs) {
            ests.push_back(run[idx].est);
            trues.push_back(double(run[idx].trueF0));
        }

        double muEst = mean(ests);
        double muTrue = mean(trues);
        double sdEst = stdev_sample(ests);

        double relBias = (muTrue > 0) ? (muEst - muTrue) / muTrue : 0.0;
        double relStd  = (muTrue > 0) ? sdEst / muTrue : 0.0;

        StatsRow r;
        r.step = idx;
        r.t = checkpoints[idx];
        r.fraction = double(checkpoints[idx]) / double(streamLen);
        r.mean_true = muTrue;
        r.mean_est = muEst;
        r.std_est = sdEst;
        r.rel_bias = relBias;
        r.rel_std = relStd;
        r.th_1_04 = th104;
        r.th_1_3 = th13;
        stats.push_back(r);
    }

    return stats;
}

static AnalysisSummary summarize_stats(const std::vector<StatsRow>& stats) {
    AnalysisSummary s;
    if (stats.empty()) return s;
    s.th104 = stats[0].th_1_04;
    s.th13 = stats[0].th_1_3;
    double sum = 0.0;
    double mx = 0.0;
    double mxBias = 0.0;
    for (const auto& r : stats) {
        sum += r.rel_std;
        mx = std::max(mx, r.rel_std);
        mxBias = std::max(mxBias, std::abs(r.rel_bias));
    }
    s.rel_std_mean = sum / double(stats.size());
    s.rel_std_max = mx;
    s.rel_bias_last = stats.back().rel_bias;
    s.rel_bias_max_abs = mxBias;
    return s;
}

static void write_stream_csv(const std::string& dir, int streamId, const std::vector<Point>& run, std::size_t streamLen) {
    std::ofstream f(dir + "/stream_" + std::to_string(streamId) + ".csv");
    f << "step,t,fraction,true_F0,est_N\n";
    for (std::size_t i = 0; i < run.size(); ++i) {
        double frac = double(run[i].t) / double(streamLen);
        f << i << "," << run[i].t << "," << std::fixed << std::setprecision(6) << frac
          << "," << run[i].trueF0 << "," << std::setprecision(6) << run[i].est << "\n";
    }
}

static void write_stats_csv(const std::string& dir, const std::vector<StatsRow>& stats) {
    std::ofstream f(dir + "/stats.csv");
    f << "step,t,fraction,mean_true,mean_est,std_est,rel_bias,rel_std,th_1_04,th_1_3\n";
    for (const auto& r : stats) {
        f << r.step << "," << r.t << "," << std::fixed << std::setprecision(6) << r.fraction
          << "," << std::setprecision(6) << r.mean_true
          << "," << std::setprecision(6) << r.mean_est
          << "," << std::setprecision(6) << r.std_est
          << "," << std::setprecision(6) << r.rel_bias
          << "," << std::setprecision(6) << r.rel_std
          << "," << std::setprecision(6) << r.th_1_04
          << "," << std::setprecision(6) << r.th_1_3
          << "\n";
    }
}

static void write_meta_and_analysis(
    const std::string& dir,
    const std::string& variant,
    int B,
    std::size_t registerMemoryBytes,
    const AnalysisSummary& sum
) {
    std::ofstream meta(dir + "/meta.txt");
    meta << "variant=" << variant << "\n";
    meta << "B=" << B << "\n";
    meta << "m=2^B=" << (1ull << B) << "\n";
    meta << "register_memory_bytes=" << registerMemoryBytes << "\n";
    meta << "theory_1.04_over_sqrt_m=" << std::setprecision(12) << sum.th104 << "\n";
    meta << "theory_1.3_over_sqrt_m=" << std::setprecision(12) << sum.th13 << "\n";

    std::ofstream a(dir + "/analysis.txt");
    a << "Проверка точности выполнена по данным stats.csv.\n";
    a << "Относительный разброс оценок по потокам в среднем: " << std::fixed << std::setprecision(4) << (sum.rel_std_mean * 100.0) << "%.\n";
    a << "Максимальный относительный разброс по точкам: " << std::fixed << std::setprecision(4) << (sum.rel_std_max * 100.0) << "%.\n";
    a << "Теоретический ориентир 1.04/sqrt(m): " << std::fixed << std::setprecision(4) << (sum.th104 * 100.0) << "%.\n";
    a << "Теоретический ориентир 1.3/sqrt(m): " << std::fixed << std::setprecision(4) << (sum.th13 * 100.0) << "%.\n";
    a << "Смещение оценки в конце потока: " << std::fixed << std::setprecision(4) << (sum.rel_bias_last * 100.0) << "%.\n";
    a << "Максимальное по модулю смещение по точкам: " << std::fixed << std::setprecision(4) << (sum.rel_bias_max_abs * 100.0) << "%.\n";
    a << "Память на регистры: " << registerMemoryBytes << " байт.\n";
}

class HyperLogLog32 {
public:
    explicit HyperLogLog32(int B) : B_(B), m_(1u << B), M_(m_, 0) {
        if (B_ < 4 || B_ > 16) throw std::runtime_error("B must be in [4..16]");
    }

    void add(std::uint32_t x) {
        std::uint32_t idx = x >> (32 - B_);
        std::uint32_t w = x << B_;
        uint8_t r = rho(w);
        if (r > M_[idx]) M_[idx] = r;
    }

    double estimate() const {
        double Z = 0.0;
        int V = 0;
        for (std::uint32_t i = 0; i < m_; ++i) {
            Z += std::ldexp(1.0, -int(M_[i]));
            if (M_[i] == 0) V++;
        }

        double E = alpha_m() * double(m_) * double(m_) / Z;

        if (E <= 2.5 * double(m_) && V > 0) {
            E = double(m_) * std::log(double(m_) / double(V));
        }

        const double two32 = 4294967296.0;
        if (E > (two32 / 30.0)) {
            double ratio = E / two32;
            if (ratio < 1.0) E = -two32 * std::log(1.0 - ratio);
        }

        return E;
    }

    std::size_t memory_bytes() const { return M_.size() * sizeof(uint8_t); }

private:
    uint8_t rho(std::uint32_t w) const {
        if (w == 0) return uint8_t((32 - B_) + 1);
        return uint8_t(__builtin_clz(w) + 1);
    }

    double alpha_m() const {
        if (m_ == 16) return 0.673;
        if (m_ == 32) return 0.697;
        if (m_ == 64) return 0.709;
        return 0.7213 / (1.0 + 1.079 / double(m_));
    }

    int B_;
    std::uint32_t m_;
    std::vector<uint8_t> M_;
};

class Packed6 {
public:
    explicit Packed6(std::size_t n) : n_(n) {
        std::size_t bits = n_ * 6;
        data_.assign((bits + 63) / 64, 0ull);
    }

    uint8_t get(std::size_t idx) const {
        std::size_t bit = idx * 6;
        std::size_t w = bit / 64;
        std::size_t off = bit % 64;

        std::uint64_t cur = data_[w] >> off;
        if (off <= 58) return uint8_t(cur & 0x3Fu);

        std::uint64_t next = data_[w + 1];
        std::uint64_t merged = cur | (next << (64 - off));
        return uint8_t(merged & 0x3Fu);
    }

    void set(std::size_t idx, uint8_t val) {
        val &= 0x3F;
        std::size_t bit = idx * 6;
        std::size_t w = bit / 64;
        std::size_t off = bit % 64;

        if (off <= 58) {
            std::uint64_t mask = (0x3Full << off);
            data_[w] = (data_[w] & ~mask) | (std::uint64_t(val) << off);
        } else {
            std::uint64_t lowBits = 64 - off;
            std::uint64_t lowMask = ((1ull << lowBits) - 1ull) << off;
            data_[w] = (data_[w] & ~lowMask) | ((std::uint64_t(val) << off) & lowMask);

            std::uint64_t highBits = 6 - lowBits;
            std::uint64_t highMask = (1ull << highBits) - 1ull;
            data_[w + 1] = (data_[w + 1] & ~highMask) | ((std::uint64_t(val) >> lowBits) & highMask);
        }
    }

    std::size_t bytes() const { return data_.size() * sizeof(std::uint64_t); }

private:
    std::size_t n_;
    std::vector<std::uint64_t> data_;
};

class HyperLogLog64Packed {
public:
    explicit HyperLogLog64Packed(int B) : B_(B), m_(1ull << B), M_(std::size_t(m_)) {
        if (B_ < 4 || B_ > 20) throw std::runtime_error("B must be in [4..20]");
    }

    void add(std::uint64_t x) {
        std::uint64_t idx = x >> (64 - B_);
        std::uint64_t w = x << B_;
        uint8_t r = rho(w);
        uint8_t cur = M_.get(std::size_t(idx));
        if (r > cur) M_.set(std::size_t(idx), r);
    }

    double estimate() const {
        double Z = 0.0;
        int V = 0;
        for (std::uint64_t i = 0; i < m_; ++i) {
            uint8_t v = M_.get(std::size_t(i));
            Z += std::ldexp(1.0, -int(v));
            if (v == 0) V++;
        }

        double E = alpha_m() * double(m_) * double(m_) / Z;

        if (E <= 2.5 * double(m_) && V > 0) {
            E = double(m_) * std::log(double(m_) / double(V));
        }

        const long double two64 = 18446744073709551616.0L;
        if (E > double(two64 / 30.0L)) {
            long double ratio = (long double)E / two64;
            if (ratio < 1.0L) E = double(-two64 * std::log(1.0L - ratio));
        }

        return E;
    }

    std::size_t memory_bytes() const { return M_.bytes(); }

private:
    uint8_t rho(std::uint64_t w) const {
        if (w == 0) return uint8_t((64 - B_) + 1);
        return uint8_t(__builtin_clzll(w) + 1);
    }

    double alpha_m() const {
        if (m_ == 16) return 0.673;
        if (m_ == 32) return 0.697;
        if (m_ == 64) return 0.709;
        return 0.7213 / (1.0 + 1.079 / double(m_));
    }

    int B_;
    std::uint64_t m_;
    Packed6 M_;
};

static std::vector<Point> run_one_stream_std(
    const std::vector<std::string>& stream,
    const std::vector<std::size_t>& checkpoints,
    const Hash64& h,
    int B
) {
    HyperLogLog32 hll(B);
    std::unordered_set<std::string> exact;
    exact.reserve(stream.size() * 2);

    std::vector<Point> out;
    out.reserve(checkpoints.size());

    std::size_t nextIdx = 0;
    for (std::size_t i = 0; i < stream.size(); ++i) {
        const std::string& s = stream[i];
        exact.insert(s);

        std::uint64_t hv = h(s);
        hll.add(std::uint32_t(hv & 0xffffffffu));

        std::size_t t = i + 1;
        while (nextIdx < checkpoints.size() && checkpoints[nextIdx] == t) {
            out.push_back(Point{t, exact.size(), hll.estimate()});
            nextIdx++;
        }
    }
    return out;
}

static std::vector<Point> run_one_stream_improved(
    const std::vector<std::string>& stream,
    const std::vector<std::size_t>& checkpoints,
    const Hash64& h,
    int B
) {
    HyperLogLog64Packed hll(B);
    std::unordered_set<std::string> exact;
    exact.reserve(stream.size() * 2);

    std::vector<Point> out;
    out.reserve(checkpoints.size());

    std::size_t nextIdx = 0;
    for (std::size_t i = 0; i < stream.size(); ++i) {
        const std::string& s = stream[i];
        exact.insert(s);

        hll.add(h(s));

        std::size_t t = i + 1;
        while (nextIdx < checkpoints.size() && checkpoints[nextIdx] == t) {
            out.push_back(Point{t, exact.size(), hll.estimate()});
            nextIdx++;
        }
    }
    return out;
}

static void write_compare(
    const std::string& outDir,
    std::size_t memStd,
    std::size_t memImp,
    const AnalysisSummary& sStd,
    const AnalysisSummary& sImp
) {
    std::ofstream f(outDir + "/compare.txt");
    f << "Сравнение стандартной и улучшенной версий.\n";
    f << "Память на регистры (стандарт): " << memStd << " байт.\n";
    f << "Память на регистры (улучшенная): " << memImp << " байт.\n";
    if (memStd > 0) {
        double save = 100.0 * (double(memStd - memImp) / double(memStd));
        f << "Экономия памяти: " << std::fixed << std::setprecision(2) << save << "%.\n";
    }
    f << "Средний относительный разброс (стандарт): " << std::fixed << std::setprecision(4) << (sStd.rel_std_mean * 100.0) << "%.\n";
    f << "Средний относительный разброс (улучшенная): " << std::fixed << std::setprecision(4) << (sImp.rel_std_mean * 100.0) << "%.\n";
    f << "Максимальный относительный разброс (стандарт): " << std::fixed << std::setprecision(4) << (sStd.rel_std_max * 100.0) << "%.\n";
    f << "Максимальный относительный разброс (улучшенная): " << std::fixed << std::setprecision(4) << (sImp.rel_std_max * 100.0) << "%.\n";
    f << "Смещение в конце потока (стандарт): " << std::fixed << std::setprecision(4) << (sStd.rel_bias_last * 100.0) << "%.\n";
    f << "Смещение в конце потока (улучшенная): " << std::fixed << std::setprecision(4) << (sImp.rel_bias_last * 100.0) << "%.\n";
}

int main(int argc, char** argv) {
    std::size_t streamLen = 200000;
    int streamsCount = 20;
    int B = 10;
    double stepPercent = 5.0;
    int maxLen = 30;
    std::uint64_t seed = 42ull;
    std::string outDir = "out";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const std::string& key) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
            return std::string(argv[++i]);
        };

        if (a == "--n") streamLen = std::stoull(need("--n"));
        else if (a == "--streams") streamsCount = std::stoi(need("--streams"));
        else if (a == "--B") B = std::stoi(need("--B"));
        else if (a == "--step") stepPercent = std::stod(need("--step"));
        else if (a == "--seed") seed = std::stoull(need("--seed"));
        else if (a == "--out") outDir = need("--out");
        else if (a == "--maxlen") maxLen = std::stoi(need("--maxlen"));
        else throw std::runtime_error("Unknown аргумент: " + a);
    }

    ensure_dir(outDir);
    ensure_dir(outDir + "/std");
    ensure_dir(outDir + "/improved");

    auto checkpoints = RandomStreamGenerator::checkpoints_by_percent(streamLen, stepPercent);

    RandomStreamGenerator gen(seed, maxLen);
    Hash64 hash(seed ^ 0xA5A5A5A5A5A5A5A5ull);

    std::vector<std::vector<Point>> stdRuns;
    std::vector<std::vector<Point>> impRuns;
    stdRuns.reserve(streamsCount);
    impRuns.reserve(streamsCount);

    for (int s = 0; s < streamsCount; ++s) {
        auto stream = gen.generate(streamLen);

        auto runStd = run_one_stream_std(stream, checkpoints, hash, B);
        auto runImp = run_one_stream_improved(stream, checkpoints, hash, B);

        stdRuns.push_back(runStd);
        impRuns.push_back(runImp);

        write_stream_csv(outDir + "/std", s, runStd, streamLen);
        write_stream_csv(outDir + "/improved", s, runImp, streamLen);
    }

    auto statsStd = build_stats(stdRuns, checkpoints, streamLen, B);
    auto statsImp = build_stats(impRuns, checkpoints, streamLen, B);

    write_stats_csv(outDir + "/std", statsStd);
    write_stats_csv(outDir + "/improved", statsImp);

    HyperLogLog32 tmpStd(B);
    HyperLogLog64Packed tmpImp(B);

    auto sumStd = summarize_stats(statsStd);
    auto sumImp = summarize_stats(statsImp);

    write_meta_and_analysis(outDir + "/std", "standard_hll_32_vector_u8", B, tmpStd.memory_bytes(), sumStd);
    write_meta_and_analysis(outDir + "/improved", "improved_hll_64_packed6", B, tmpImp.memory_bytes(), sumImp);

    write_compare(outDir, tmpStd.memory_bytes(), tmpImp.memory_bytes(), sumStd, sumImp);

    std::cout << "Готово.\n";
    std::cout << "out/std: stream_*.csv, stats.csv, meta.txt, analysis.txt\n";
    std::cout << "out/improved: stream_*.csv, stats.csv, meta.txt, analysis.txt\n";
    std::cout << "out/compare.txt\n";
    return 0;
}
