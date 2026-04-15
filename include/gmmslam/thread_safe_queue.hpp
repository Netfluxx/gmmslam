#pragma once
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

namespace gmmslam {

template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(std::size_t max_size = 0)
        : max_size_(max_size) {}

    bool tryPush(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (max_size_ > 0 && queue_.size() >= max_size_) {
            return false;
        }
        queue_.push(std::move(item));
        cond_.notify_one();
        return true;
    }

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (max_size_ > 0) {
            cond_full_.wait(lock, [this] { return queue_.size() < max_size_; });
        }
        queue_.push(std::move(item));
        cond_.notify_one();
    }

    std::optional<T> tryPop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        cond_full_.notify_one();
        return item;
    }

    std::optional<T> popWithTimeout(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        cond_full_.notify_one();
        return item;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        queue_.swap(empty);
        cond_full_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    std::condition_variable cond_full_;
    std::queue<T> queue_;
    std::size_t max_size_;
};

} // namespace gmmslam
