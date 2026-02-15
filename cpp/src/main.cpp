#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace native_engine {

enum class Action { None, Buy, Exit };

struct Signal {
    Action action{Action::None};
    double confidence{0.0};
};

struct Candle {
    std::string symbol;
    std::int64_t timestamp_ms{0};
    double open{0.0};
    double high{0.0};
    double low{0.0};
    double close{0.0};
    double volume{0.0};
    double vwap{0.0};
};

struct Position {
    std::string symbol;
    double quantity{0.0};
    double entry_price{0.0};
    std::int64_t entry_time{0};
    double commission_paid{0.0};
    double current_price{0.0};
    bool is_open{true};
};

struct Trade {
    std::string trade_id;
    std::string symbol;
    std::string side;
    double quantity{0.0};
    double price{0.0};
    std::int64_t timestamp_ms{0};
    double commission{0.0};
    std::string strategy_id;
    std::string signal_id;
    double realized_pnl{0.0};
};

struct Portfolio {
    double starting_cash{0.0};
    double cash{0.0};
    double equity{0.0};
    std::unordered_map<std::string, Position> positions;
    std::vector<Trade> trades;
};

struct EquityPoint {
    std::int64_t timestamp_ms{0};
    double equity{0.0};
    double cash{0.0};
};

struct SelectionResult {
    std::string symbol;
    double score{0.0};
    double volume_surge{0.0};
    double volatility_expansion{0.0};
    double momentum{0.0};
    double breakout{0.0};
    double relative_strength{0.0};
};

struct Config {
    std::string strategy_id{"momentum_vwap_v1"};
    std::string mode{"simulate"};
    double commission_rate{0.001};
    double max_position_value{50.0};
    double risk_fraction{0.25};
    double starting_cash{100.0};
    int max_positions{5};
    int lookback_days{20};
    int recent_window{5};
    int prior_window{10};
    int breakout_window{20};
    int max_symbols{5};
    double weight_volume_surge{0.2};
    double weight_volatility_expansion{0.2};
    double weight_momentum{0.2};
    double weight_breakout{0.2};
    double weight_relative_strength{0.2};
};

struct SymbolHistory {
    std::vector<double> closes;
    std::vector<double> highs;
    std::vector<double> lows;
    std::vector<double> volumes;
    double cum_pv{0.0};
    double cum_vol{0.0};
    int last_day{0};
};

static std::string trim(const std::string &value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

static std::vector<std::string> split(const std::string &line, char delimiter) {
    std::vector<std::string> result;
    std::string token;
    std::stringstream ss(line);
    while (std::getline(ss, token, delimiter)) {
        result.push_back(trim(token));
    }
    return result;
}

static int day_key_from_ms(std::int64_t timestamp_ms) {
    std::time_t secs = static_cast<std::time_t>(timestamp_ms / 1000);
    std::tm *tm_ptr = std::gmtime(&secs);
    if (!tm_ptr) {
        return 0;
    }
    return (tm_ptr->tm_year + 1900) * 10000 + (tm_ptr->tm_mon + 1) * 100 + tm_ptr->tm_mday;
}

static double mean_last(const std::vector<double> &values, int window) {
    if (static_cast<int>(values.size()) < window) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (int i = static_cast<int>(values.size()) - window; i < static_cast<int>(values.size()); ++i) {
        sum += values[i];
    }
    return sum / window;
}

static double std_last(const std::vector<double> &values, int window) {
    if (static_cast<int>(values.size()) < window) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double mean = mean_last(values, window);
    double sum_sq = 0.0;
    for (int i = static_cast<int>(values.size()) - window; i < static_cast<int>(values.size()); ++i) {
        double diff = values[i] - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / window);
}

static double max_last(const std::vector<double> &values, int window) {
    if (static_cast<int>(values.size()) < window) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double max_val = values[values.size() - window];
    for (int i = static_cast<int>(values.size()) - window; i < static_cast<int>(values.size()); ++i) {
        max_val = std::max(max_val, values[i]);
    }
    return max_val;
}

static double min_last(const std::vector<double> &values, int window) {
    if (static_cast<int>(values.size()) < window) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double min_val = values[values.size() - window];
    for (int i = static_cast<int>(values.size()) - window; i < static_cast<int>(values.size()); ++i) {
        min_val = std::min(min_val, values[i]);
    }
    return min_val;
}

static double rsi_last(const std::vector<double> &values, int period) {
    if (static_cast<int>(values.size()) < period + 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double gains = 0.0;
    double losses = 0.0;
    for (int i = static_cast<int>(values.size()) - period; i < static_cast<int>(values.size()); ++i) {
        double diff = values[i] - values[i - 1];
        if (diff > 0) {
            gains += diff;
        } else {
            losses -= diff;
        }
    }
    if (losses == 0 && gains == 0) {
        return 50.0;
    }
    if (losses == 0) {
        return 100.0;
    }
    if (gains == 0) {
        return 0.0;
    }
    double rs = gains / losses;
    return 100.0 - (100.0 / (1.0 + rs));
}

static double mean_range(const std::vector<double> &values, int start, int length) {
    if (start < 0 || length <= 0 || start + length > static_cast<int>(values.size())) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (int i = start; i < start + length; ++i) {
        sum += values[i];
    }
    return sum / length;
}

static double std_range(const std::vector<double> &values, int start, int length) {
    if (start < 0 || length <= 1 || start + length > static_cast<int>(values.size())) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double mean = mean_range(values, start, length);
    double sum_sq = 0.0;
    for (int i = start; i < start + length; ++i) {
        double diff = values[i] - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / length);
}

static double max_range(const std::vector<double> &values, int start, int length) {
    if (start < 0 || length <= 0 || start + length > static_cast<int>(values.size())) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double max_val = values[start];
    for (int i = start; i < start + length; ++i) {
        max_val = std::max(max_val, values[i]);
    }
    return max_val;
}

class Strategy {
  public:
    virtual ~Strategy() = default;
    virtual std::string id() const = 0;
    virtual Signal on_candle(const std::string &symbol, SymbolHistory &history, const Candle &candle) = 0;
};

class MomentumVwapStrategy : public Strategy {
  public:
    std::string id() const override { return "momentum_vwap_v1"; }

    Signal on_candle(const std::string &symbol, SymbolHistory &history, const Candle &candle) override {
        (void)symbol;
        const int volume_window = 20;
        const int momentum_window = 5;
        const double vwap_threshold = 0.001;
        const double exit_threshold = 0.0;
        if (static_cast<int>(history.closes.size()) < std::max(volume_window, momentum_window) + 2) {
            return {};
        }

        double vwap = history.cum_vol > 0 ? history.cum_pv / history.cum_vol : 0.0;
        if (vwap <= 0) {
            return {};
        }

        double volume_ma = mean_last(history.volumes, volume_window);
        double momentum_prev = (history.closes[history.closes.size() - momentum_window - 1] == 0)
                                  ? 0.0
                                  : (history.closes.back() / history.closes[history.closes.size() - momentum_window - 1] -
                                     1.0);
        double momentum_now = (history.closes[history.closes.size() - momentum_window] == 0)
                                 ? 0.0
                                 : (history.closes.back() / history.closes[history.closes.size() - momentum_window] - 1.0);

        double prev_close = history.closes[history.closes.size() - 2];
        bool crossed_up = prev_close <= vwap * (1 + vwap_threshold) && candle.close > vwap * (1 + vwap_threshold);
        bool crossed_down = prev_close >= vwap * (1 + exit_threshold) && candle.close < vwap * (1 + exit_threshold);
        bool volume_confirm = std::isnan(volume_ma) || volume_ma <= 0 ? true : candle.volume > volume_ma;
        bool momentum_positive = momentum_now > 0;
        bool momentum_cross_down = momentum_prev > 0 && momentum_now <= 0;

        if (crossed_up && volume_confirm && momentum_positive) {
            return {Action::Buy, 0.6};
        }
        if (crossed_down || momentum_cross_down) {
            return {Action::Exit, 0.5};
        }
        return {};
    }
};

class RsiMeanReversionStrategy : public Strategy {
  public:
    std::string id() const override { return "rsi_mean_reversion_v1"; }

    Signal on_candle(const std::string &symbol, SymbolHistory &history, const Candle &candle) override {
        (void)symbol;
        (void)candle;
        const int period = 14;
        const double oversold = 30.0;
        const double exit_level = 50.0;
        if (static_cast<int>(history.closes.size()) < period + 2) {
            return {};
        }
        double prev_rsi = rsi_last(std::vector<double>(history.closes.begin(), history.closes.end() - 1), period);
        double current_rsi = rsi_last(history.closes, period);
        if (std::isnan(prev_rsi) || std::isnan(current_rsi)) {
            return {};
        }
        bool crossed_oversold = prev_rsi < oversold && current_rsi >= oversold;
        bool crossed_exit = prev_rsi < exit_level && current_rsi >= exit_level;
        if (crossed_oversold) {
            return {Action::Buy, 0.55};
        }
        if (crossed_exit) {
            return {Action::Exit, 0.5};
        }
        return {};
    }
};

class EmaCrossoverStrategy : public Strategy {
  public:
    std::string id() const override { return "ema_crossover_v1"; }

    Signal on_candle(const std::string &symbol, SymbolHistory &history, const Candle &candle) override {
        (void)symbol;
        const int fast_period = 12;
        const int slow_period = 26;
        const double min_spread = 0.0005;
        auto &state = states_[symbol];
        if (!state.initialized) {
            state.ema_fast = candle.close;
            state.ema_slow = candle.close;
            state.prev_ema_fast = candle.close;
            state.prev_ema_slow = candle.close;
            state.initialized = true;
            return {};
        }
        state.prev_ema_fast = state.ema_fast;
        state.prev_ema_slow = state.ema_slow;
        double alpha_fast = 2.0 / (fast_period + 1.0);
        double alpha_slow = 2.0 / (slow_period + 1.0);
        state.ema_fast = alpha_fast * candle.close + (1 - alpha_fast) * state.ema_fast;
        state.ema_slow = alpha_slow * candle.close + (1 - alpha_slow) * state.ema_slow;

        if (state.ema_slow == 0 || state.prev_ema_slow == 0) {
            return {};
        }
        double spread_now = (state.ema_fast - state.ema_slow) / state.ema_slow;
        double spread_prev = (state.prev_ema_fast - state.prev_ema_slow) / state.prev_ema_slow;
        bool crossed_up = spread_prev <= min_spread && spread_now > min_spread;
        bool crossed_down = spread_prev >= -min_spread && spread_now < -min_spread;
        if (crossed_up) {
            return {Action::Buy, 0.6};
        }
        if (crossed_down) {
            return {Action::Exit, 0.5};
        }
        return {};
    }

  private:
    struct EmaState {
        bool initialized{false};
        double ema_fast{0.0};
        double ema_slow{0.0};
        double prev_ema_fast{0.0};
        double prev_ema_slow{0.0};
    };
    std::unordered_map<std::string, EmaState> states_;
};

class BollingerReversionStrategy : public Strategy {
  public:
    std::string id() const override { return "bollinger_reversion_v1"; }

    Signal on_candle(const std::string &symbol, SymbolHistory &history, const Candle &candle) override {
        (void)symbol;
        (void)candle;
        const int window = 20;
        const double std_dev = 2.0;
        if (static_cast<int>(history.closes.size()) < window + 2) {
            return {};
        }
        double sma = mean_last(history.closes, window);
        double std = std_last(history.closes, window);
        if (std::isnan(sma) || std::isnan(std)) {
            return {};
        }
        double lower = sma - std_dev * std;
        double upper = sma + std_dev * std;
        (void)upper;
        double prev_close = history.closes[history.closes.size() - 2];
        bool crossed_up = prev_close <= lower && history.closes.back() > lower;
        bool crossed_mid = prev_close <= sma && history.closes.back() > sma;
        if (crossed_up) {
            return {Action::Buy, 0.55};
        }
        if (crossed_mid) {
            return {Action::Exit, 0.5};
        }
        return {};
    }
};

class DonchianBreakoutStrategy : public Strategy {
  public:
    std::string id() const override { return "donchian_breakout_v1"; }

    Signal on_candle(const std::string &symbol, SymbolHistory &history, const Candle &candle) override {
        (void)symbol;
        (void)candle;
        const int entry_window = 20;
        const int exit_window = 10;
        if (static_cast<int>(history.highs.size()) < entry_window + 2 ||
            static_cast<int>(history.lows.size()) < exit_window + 2) {
            return {};
        }
        std::vector<double> highs(history.highs.begin(), history.highs.end() - 1);
        std::vector<double> lows(history.lows.begin(), history.lows.end() - 1);
        double entry_high = max_last(highs, entry_window);
        double exit_low = min_last(lows, exit_window);
        if (std::isnan(entry_high) || std::isnan(exit_low)) {
            return {};
        }
        double prev_close = history.closes[history.closes.size() - 2];
        bool crossed_up = prev_close <= entry_high && history.closes.back() > entry_high;
        bool crossed_down = prev_close >= exit_low && history.closes.back() < exit_low;
        if (crossed_up) {
            return {Action::Buy, 0.6};
        }
        if (crossed_down) {
            return {Action::Exit, 0.5};
        }
        return {};
    }
};

class StrategyFactory {
  public:
    static std::unique_ptr<Strategy> create(const std::string &id) {
        if (id == "momentum_vwap_v1") {
            return std::make_unique<MomentumVwapStrategy>();
        }
        if (id == "rsi_mean_reversion_v1") {
            return std::make_unique<RsiMeanReversionStrategy>();
        }
        if (id == "ema_crossover_v1") {
            return std::make_unique<EmaCrossoverStrategy>();
        }
        if (id == "bollinger_reversion_v1") {
            return std::make_unique<BollingerReversionStrategy>();
        }
        if (id == "donchian_breakout_v1") {
            return std::make_unique<DonchianBreakoutStrategy>();
        }
        return nullptr;
    }
};

class Engine {
  public:
    Engine(Config config, std::unique_ptr<Strategy> strategy)
        : config_(std::move(config)), strategy_(std::move(strategy)) {}

    std::pair<Portfolio, std::vector<EquityPoint>> run(
        const std::unordered_map<std::string, std::vector<Candle>> &candles_by_symbol) {
        Portfolio portfolio{config_.starting_cash, config_.starting_cash, config_.starting_cash, {}, {}};
        std::vector<EquityPoint> equity_curve;
        std::unordered_map<std::string, SymbolHistory> histories;

        using HeapItem = std::tuple<std::int64_t, std::string, std::size_t>;
        auto cmp = [](const HeapItem &a, const HeapItem &b) { return std::get<0>(a) > std::get<0>(b); };
        std::priority_queue<HeapItem, std::vector<HeapItem>, decltype(cmp)> heap(cmp);

        for (const auto &entry : candles_by_symbol) {
            if (!entry.second.empty()) {
                heap.emplace(entry.second.front().timestamp_ms, entry.first, 0);
            }
        }

        if (heap.empty()) {
            return {portfolio, equity_curve};
        }

        while (!heap.empty()) {
            auto [current_ts, symbol, index] = heap.top();
            heap.pop();
            std::vector<HeapItem> batch;
            batch.emplace_back(current_ts, symbol, index);
            while (!heap.empty() && std::get<0>(heap.top()) == current_ts) {
                batch.push_back(heap.top());
                heap.pop();
            }

            for (const auto &item : batch) {
                const auto &sym = std::get<1>(item);
                std::size_t idx = std::get<2>(item);
                const auto &candle = candles_by_symbol.at(sym)[idx];
                auto &history = histories[sym];
                update_history(history, candle);
                mark_to_market(portfolio, sym, candle);
                Signal signal = strategy_->on_candle(sym, history, candle);
                process_signal(portfolio, signal, candle);

                std::size_t next_idx = idx + 1;
                if (next_idx < candles_by_symbol.at(sym).size()) {
                    heap.emplace(candles_by_symbol.at(sym)[next_idx].timestamp_ms, sym, next_idx);
                }
            }

            portfolio.equity = calculate_equity(portfolio);
            equity_curve.push_back({current_ts, portfolio.equity, portfolio.cash});
        }

        return {portfolio, equity_curve};
    }

  private:
    void update_history(SymbolHistory &history, const Candle &candle) {
        int day_key = day_key_from_ms(candle.timestamp_ms);
        if (history.last_day != day_key) {
            history.cum_pv = 0.0;
            history.cum_vol = 0.0;
            history.last_day = day_key;
        }
        history.closes.push_back(candle.close);
        history.highs.push_back(candle.high);
        history.lows.push_back(candle.low);
        history.volumes.push_back(candle.volume);
        history.cum_pv += candle.close * candle.volume;
        history.cum_vol += candle.volume;
    }

    void mark_to_market(Portfolio &portfolio, const std::string &symbol, const Candle &candle) {
        auto it = portfolio.positions.find(symbol);
        if (it == portfolio.positions.end()) {
            return;
        }
        it->second.current_price = candle.close;
    }

    void process_signal(Portfolio &portfolio, const Signal &signal, const Candle &candle) {
        if (signal.action == Action::Buy) {
            enter_position(portfolio, signal, candle);
        } else if (signal.action == Action::Exit) {
            exit_position(portfolio, candle);
        }
    }

    void enter_position(Portfolio &portfolio, const Signal &signal, const Candle &candle) {
        if (portfolio.positions.find(candle.symbol) != portfolio.positions.end()) {
            return;
        }
        if (static_cast<int>(portfolio.positions.size()) >= config_.max_positions) {
            return;
        }
        double risk_budget = portfolio.cash * config_.risk_fraction;
        double target_value = std::min(risk_budget * std::max(signal.confidence, 0.0), config_.max_position_value);
        if (candle.close <= 0) {
            return;
        }
        double quantity = target_value / candle.close;
        if (quantity <= 0) {
            return;
        }
        double max_affordable = portfolio.cash / (candle.close * (1 + config_.commission_rate));
        quantity = std::min(quantity, max_affordable);
        if (quantity <= 0) {
            return;
        }
        double notional = quantity * candle.close;
        double commission = notional * config_.commission_rate;
        double total_cost = notional + commission;
        if (total_cost > portfolio.cash) {
            return;
        }
        Position position{candle.symbol, quantity, candle.close, candle.timestamp_ms, commission, candle.close, true};
        portfolio.positions[candle.symbol] = position;
        portfolio.cash -= total_cost;

        Trade trade;
        trade.trade_id = "trd_" + candle.symbol + "_" + std::to_string(++trade_counter_);
        trade.symbol = candle.symbol;
        trade.side = "buy";
        trade.quantity = quantity;
        trade.price = candle.close;
        trade.timestamp_ms = candle.timestamp_ms;
        trade.commission = commission;
        trade.strategy_id = strategy_->id();
        trade.signal_id = "sig_" + candle.symbol + "_" + std::to_string(trade_counter_);
        trade.realized_pnl = 0.0;
        portfolio.trades.push_back(trade);
    }

    void exit_position(Portfolio &portfolio, const Candle &candle) {
        auto it = portfolio.positions.find(candle.symbol);
        if (it == portfolio.positions.end()) {
            return;
        }
        Position &position = it->second;
        double quantity = position.quantity;
        double entry_price = position.entry_price;
        double entry_commission = position.commission_paid;
        double notional = quantity * candle.close;
        double commission = notional * config_.commission_rate;
        double realized = (candle.close - entry_price) * quantity;
        realized -= (commission + entry_commission);

        portfolio.cash += notional - commission;

        Trade trade;
        trade.trade_id = "trd_" + candle.symbol + "_" + std::to_string(++trade_counter_);
        trade.symbol = candle.symbol;
        trade.side = "sell";
        trade.quantity = quantity;
        trade.price = candle.close;
        trade.timestamp_ms = candle.timestamp_ms;
        trade.commission = commission;
        trade.strategy_id = strategy_->id();
        trade.signal_id = "sig_" + candle.symbol + "_" + std::to_string(trade_counter_);
        trade.realized_pnl = realized;
        portfolio.trades.push_back(trade);
        portfolio.positions.erase(it);
    }

    double calculate_equity(const Portfolio &portfolio) const {
        double open_value = 0.0;
        for (const auto &entry : portfolio.positions) {
            open_value += entry.second.quantity * entry.second.current_price;
        }
        return portfolio.cash + open_value;
    }

    Config config_;
    std::unique_ptr<Strategy> strategy_;
    std::int64_t trade_counter_{0};
};

static bool compute_selection_metrics(
    const std::vector<Candle> &candles,
    const Config &config,
    SelectionResult &result) {
    int n = static_cast<int>(candles.size());
    int min_len = std::max({config.lookback_days, config.prior_window + config.recent_window + 1,
                            config.breakout_window + 1});
    if (n < min_len) {
        return false;
    }

    std::vector<double> closes;
    std::vector<double> highs;
    std::vector<double> volumes;
    closes.reserve(n);
    highs.reserve(n);
    volumes.reserve(n);
    for (const auto &candle : candles) {
        closes.push_back(candle.close);
        highs.push_back(candle.high);
        volumes.push_back(candle.volume);
    }

    int recent_start = n - config.recent_window;
    int prior_start = n - config.recent_window - config.prior_window;
    double recent_vol = mean_range(volumes, recent_start, config.recent_window);
    double prior_vol = mean_range(volumes, prior_start, config.prior_window);
    double volume_surge = (prior_vol > 0.0) ? (recent_vol / prior_vol) : 0.0;

    std::vector<double> returns;
    returns.reserve(n - 1);
    for (int i = 1; i < n; ++i) {
        if (closes[i - 1] == 0.0) {
            returns.push_back(0.0);
        } else {
            returns.push_back(closes[i] / closes[i - 1] - 1.0);
        }
    }
    int m = static_cast<int>(returns.size());
    if (m < config.recent_window + config.prior_window) {
        return false;
    }
    int recent_ret_start = m - config.recent_window;
    int prior_ret_start = m - config.recent_window - config.prior_window;
    double recent_std = std_range(returns, recent_ret_start, config.recent_window);
    double prior_std = std_range(returns, prior_ret_start, config.prior_window);
    double volatility_expansion = (prior_std > 0.0 && !std::isnan(prior_std)) ? (recent_std / prior_std) : 0.0;

    double past_close = closes[n - 1 - config.prior_window];
    double momentum = past_close != 0.0 ? (closes.back() / past_close - 1.0) : 0.0;

    double prior_high = max_range(highs, n - 1 - config.breakout_window, config.breakout_window);
    double breakout = prior_high > 0.0 ? (closes.back() / prior_high - 1.0) : 0.0;

    result.volume_surge = std::isnan(volume_surge) ? 0.0 : volume_surge;
    result.volatility_expansion = std::isnan(volatility_expansion) ? 0.0 : volatility_expansion;
    result.momentum = std::isnan(momentum) ? 0.0 : momentum;
    result.breakout = std::isnan(breakout) ? 0.0 : breakout;
    return true;
}

static std::vector<SelectionResult> select_symbols(
    const std::unordered_map<std::string, std::vector<Candle>> &candles_by_symbol,
    const Config &config) {
    std::unordered_map<std::string, SelectionResult> metrics;
    metrics.reserve(candles_by_symbol.size());

    for (const auto &entry : candles_by_symbol) {
        SelectionResult result;
        result.symbol = entry.first;
        if (compute_selection_metrics(entry.second, config, result)) {
            metrics[entry.first] = result;
        }
    }

    if (metrics.empty()) {
        return {};
    }

    std::vector<std::pair<std::string, double>> momentum_pairs;
    momentum_pairs.reserve(metrics.size());
    for (const auto &entry : metrics) {
        momentum_pairs.emplace_back(entry.first, entry.second.momentum);
    }
    std::sort(momentum_pairs.begin(), momentum_pairs.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
    std::unordered_map<std::string, double> relative_strength;
    int n = static_cast<int>(momentum_pairs.size());
    int idx = 0;
    while (idx < n) {
        int j = idx;
        while (j + 1 < n && momentum_pairs[j + 1].second == momentum_pairs[idx].second) {
            ++j;
        }
        double rank = (idx + j + 2) / 2.0;
        double pct_rank = rank / n;
        for (int k = idx; k <= j; ++k) {
            relative_strength[momentum_pairs[k].first] = pct_rank;
        }
        idx = j + 1;
    }

    double min_vs = std::numeric_limits<double>::infinity();
    double max_vs = -std::numeric_limits<double>::infinity();
    double min_vol = std::numeric_limits<double>::infinity();
    double max_vol = -std::numeric_limits<double>::infinity();
    double min_mom = std::numeric_limits<double>::infinity();
    double max_mom = -std::numeric_limits<double>::infinity();
    double min_break = std::numeric_limits<double>::infinity();
    double max_break = -std::numeric_limits<double>::infinity();

    for (const auto &entry : metrics) {
        const auto &m = entry.second;
        min_vs = std::min(min_vs, m.volume_surge);
        max_vs = std::max(max_vs, m.volume_surge);
        min_vol = std::min(min_vol, m.volatility_expansion);
        max_vol = std::max(max_vol, m.volatility_expansion);
        min_mom = std::min(min_mom, m.momentum);
        max_mom = std::max(max_mom, m.momentum);
        min_break = std::min(min_break, m.breakout);
        max_break = std::max(max_break, m.breakout);
    }

    std::vector<SelectionResult> results;
    results.reserve(metrics.size());
    for (auto &entry : metrics) {
        SelectionResult result = entry.second;
        auto norm = [](double value, double min_val, double max_val) {
            if (max_val == min_val) {
                return 0.0;
            }
            return (value - min_val) / (max_val - min_val);
        };
        double volume_norm = norm(result.volume_surge, min_vs, max_vs);
        double volatility_norm = norm(result.volatility_expansion, min_vol, max_vol);
        double momentum_norm = norm(result.momentum, min_mom, max_mom);
        double breakout_norm = norm(result.breakout, min_break, max_break);
        double rel_strength = relative_strength[result.symbol];
        result.relative_strength = rel_strength;
        result.score = volume_norm * config.weight_volume_surge +
                       volatility_norm * config.weight_volatility_expansion +
                       momentum_norm * config.weight_momentum +
                       breakout_norm * config.weight_breakout +
                       rel_strength * config.weight_relative_strength;
        results.push_back(result);
    }

    std::sort(results.begin(), results.end(),
              [](const SelectionResult &a, const SelectionResult &b) { return a.score > b.score; });
    if (static_cast<int>(results.size()) > config.max_symbols) {
        results.resize(config.max_symbols);
    }
    return results;
}

static void write_selection(const std::string &path, const std::vector<SelectionResult> &results) {
    std::ofstream file(path);
    file << "symbol,score,volume_surge,volatility_expansion,momentum,breakout,relative_strength\n";
    for (const auto &row : results) {
        file << row.symbol << ',' << row.score << ',' << row.volume_surge << ',' << row.volatility_expansion << ','
             << row.momentum << ',' << row.breakout << ',' << row.relative_strength << '\n';
    }
}

static bool load_config(const std::string &path, Config &config) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << path << '\n';
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        auto pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));
        try {
            if (key == "strategy_id") {
                config.strategy_id = value;
            } else if (key == "mode") {
                config.mode = value;
            } else if (key == "commission_rate") {
                config.commission_rate = std::stod(value);
            } else if (key == "max_position_value") {
                config.max_position_value = std::stod(value);
            } else if (key == "risk_fraction") {
                config.risk_fraction = std::stod(value);
            } else if (key == "starting_cash") {
                config.starting_cash = std::stod(value);
            } else if (key == "max_positions") {
                config.max_positions = std::stoi(value);
            } else if (key == "lookback_days") {
                config.lookback_days = std::stoi(value);
            } else if (key == "recent_window") {
                config.recent_window = std::stoi(value);
            } else if (key == "prior_window") {
                config.prior_window = std::stoi(value);
            } else if (key == "breakout_window") {
                config.breakout_window = std::stoi(value);
            } else if (key == "max_symbols") {
                config.max_symbols = std::stoi(value);
            } else if (key == "weight_volume_surge") {
                config.weight_volume_surge = std::stod(value);
            } else if (key == "weight_volatility_expansion") {
                config.weight_volatility_expansion = std::stod(value);
            } else if (key == "weight_momentum") {
                config.weight_momentum = std::stod(value);
            } else if (key == "weight_breakout") {
                config.weight_breakout = std::stod(value);
            } else if (key == "weight_relative_strength") {
                config.weight_relative_strength = std::stod(value);
            }
        } catch (const std::exception &ex) {
            std::cerr << "Config parse error for key '" << key << "': " << ex.what() << '\n';
            return false;
        }
    }
    return true;
}

static std::optional<std::string> validate_config(const Config &config) {
    if (config.mode != "simulate" && config.mode != "select") {
        return "mode must be 'simulate' or 'select'";
    }
    if (config.starting_cash <= 0) {
        return "starting_cash must be > 0";
    }
    if (config.risk_fraction <= 0.0 || config.risk_fraction > 1.0) {
        return "risk_fraction must be in (0, 1]";
    }
    if (config.max_position_value <= 0.0) {
        return "max_position_value must be > 0";
    }
    if (config.max_positions <= 0) {
        return "max_positions must be > 0";
    }
    if (config.commission_rate < 0.0) {
        return "commission_rate must be >= 0";
    }
    if (config.lookback_days <= 0) {
        return "lookback_days must be > 0";
    }
    if (config.recent_window <= 0 || config.prior_window <= 0 || config.breakout_window <= 0) {
        return "recent_window, prior_window, breakout_window must be > 0";
    }
    if (config.max_symbols <= 0) {
        return "max_symbols must be > 0";
    }
    return std::nullopt;
}

static bool load_candles(
    const std::string &path,
    std::unordered_map<std::string, std::vector<Candle>> &candles_by_symbol,
    std::string *error) {
    std::ifstream file(path);
    if (!file.is_open()) {
        if (error) {
            *error = "Failed to open candles file: " + path;
        }
        return false;
    }
    std::string header;
    if (!std::getline(file, header)) {
        if (error) {
            *error = "Candles file missing header";
        }
        return false;
    }
    int bad_lines = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        auto fields = split(line, ',');
        if (fields.size() < 7) {
            ++bad_lines;
            continue;
        }
        try {
            Candle candle;
            candle.symbol = fields[0];
            candle.timestamp_ms = std::stoll(fields[1]);
            candle.open = std::stod(fields[2]);
            candle.high = std::stod(fields[3]);
            candle.low = std::stod(fields[4]);
            candle.close = std::stod(fields[5]);
            candle.volume = std::stod(fields[6]);
            if (fields.size() > 7 && !fields[7].empty()) {
                candle.vwap = std::stod(fields[7]);
            }
            candles_by_symbol[candle.symbol].push_back(candle);
        } catch (const std::exception &) {
            ++bad_lines;
            continue;
        }
    }
    for (auto &entry : candles_by_symbol) {
        std::sort(entry.second.begin(), entry.second.end(),
                  [](const Candle &a, const Candle &b) { return a.timestamp_ms < b.timestamp_ms; });
    }
    if (candles_by_symbol.empty()) {
        if (error) {
            *error = "No candle rows loaded";
        }
        return false;
    }
    if (bad_lines > 0) {
        std::cerr << "Skipped " << bad_lines << " malformed candle rows\n";
    }
    return true;
}

static void write_trades(const std::string &path, const std::vector<Trade> &trades) {
    std::ofstream file(path);
    file << "trade_id,symbol,side,quantity,price,timestamp,commission,strategy_id,signal_id,realized_pnl,notional\n";
    for (const auto &trade : trades) {
        file << trade.trade_id << ',' << trade.symbol << ',' << trade.side << ',' << trade.quantity << ','
             << trade.price << ',' << trade.timestamp_ms << ',' << trade.commission << ',' << trade.strategy_id << ','
             << trade.signal_id << ',' << trade.realized_pnl << ',' << trade.quantity * trade.price << '\n';
    }
}

static void write_equity(const std::string &path, const std::vector<EquityPoint> &equity_curve) {
    std::ofstream file(path);
    file << "timestamp,equity,cash\n";
    for (const auto &point : equity_curve) {
        file << point.timestamp_ms << ',' << point.equity << ',' << point.cash << '\n';
    }
}

}  // namespace native_engine

int main(int argc, char **argv) {
    using namespace native_engine;

    std::string config_path;
    std::string candles_path;
    std::string output_dir;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--candles" && i + 1 < argc) {
            candles_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        }
    }

    if (config_path.empty() || candles_path.empty() || output_dir.empty()) {
        std::cerr << "Usage: native_engine --config <config> --candles <candles.csv> --output <output_dir>\n";
        return 1;
    }

    Config config;
    if (!load_config(config_path, config)) {
        std::cerr << "Failed to load config\n";
        return 1;
    }
    if (auto validation_error = validate_config(config)) {
        std::cerr << "Config validation failed: " << *validation_error << '\n';
        return 1;
    }

    std::unordered_map<std::string, std::vector<Candle>> candles_by_symbol;
    std::string candles_error;
    if (!load_candles(candles_path, candles_by_symbol, &candles_error)) {
        std::cerr << (candles_error.empty() ? "Failed to load candles" : candles_error) << '\n';
        return 1;
    }

    if (config.mode == "select") {
        auto selection = select_symbols(candles_by_symbol, config);
        std::string selection_path = output_dir + "/selection.csv";
        write_selection(selection_path, selection);
        return 0;
    }

    auto strategy = StrategyFactory::create(config.strategy_id);
    if (!strategy) {
        std::cerr << "Unknown strategy id: " << config.strategy_id << '\n';
        return 1;
    }

    Engine engine(config, std::move(strategy));
    auto [portfolio, equity_curve] = engine.run(candles_by_symbol);

    std::string trades_path = output_dir + "/trades.csv";
    std::string equity_path = output_dir + "/equity.csv";
    write_trades(trades_path, portfolio.trades);
    write_equity(equity_path, equity_curve);

    return 0;
}
