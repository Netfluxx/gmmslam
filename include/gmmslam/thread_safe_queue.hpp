#pragma once
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

namespace gmmslam {

template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(std::size_t max_size = 0)
        : max_size_(max_size),
          ring_(max_size > 0 ? max_size : 0) {}

    bool tryPush(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (max_size_ > 0 && size_ >= max_size_) {
            return false;
        }
        pushUnlocked(std::move(item));
        cond_.notify_one();
        return true;
    }

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (max_size_ > 0) {
            cond_full_.wait(lock, [this] { return size_ < max_size_; });
        }
        pushUnlocked(std::move(item));
        cond_.notify_one();
    }

    std::optional<T> tryPop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (size_ == 0) {
            return std::nullopt;
        }
        T item = popUnlocked();
        cond_full_.notify_one();
        return item;
    }

    std::optional<T> popWithTimeout(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_.wait_for(lock, timeout, [this] { return size_ > 0; })) {
            return std::nullopt;
        }
        T item = popUnlocked();
        cond_full_.notify_one();
        return item;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_ == 0;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (max_size_ > 0) {
            for (std::size_t i = 0; i < size_; ++i) {
                ring_[(head_ + i) % max_size_].reset();
            }
            head_ = 0;
            tail_ = 0;
            size_ = 0;
        } else {
            overflow_.clear();
            size_ = 0;
        }
        cond_full_.notify_all();
    }

private:
    void pushUnlocked(T item) {
        if (max_size_ > 0) {
            ring_[tail_].reset();
            ring_[tail_].emplace(std::move(item));
            tail_ = (tail_ + 1) % max_size_;
            ++size_;
        } else {
            overflow_.push_back(std::move(item));
            size_ = overflow_.size();
        }
    }

    T popUnlocked() {
        if (max_size_ > 0) {
            std::optional<T>& slot = ring_[head_];
            T item = std::move(*slot);
            slot.reset();
            head_ = (head_ + 1) % max_size_;
            --size_;
            return item;
        }

        T item = std::move(overflow_.front());
        overflow_.pop_front();
        size_ = overflow_.size();
        return item;
    }

    mutable std::mutex mutex_;
    std::condition_variable cond_;
    std::condition_variable cond_full_;
    std::size_t max_size_;
    std::vector<std::optional<T>> ring_;
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t size_ = 0;
    std::deque<T> overflow_;
};

} // namespace gmmslam
