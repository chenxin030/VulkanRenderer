#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool
{
public:
    ThreadPool() = default;
    explicit ThreadPool(uint32_t workerCount);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void start(uint32_t workerCount);
    void stop();

    [[nodiscard]] uint32_t workerCount() const;

    template <typename Fn, typename... Args>
    auto enqueue(Fn&& fn, Args&&... args) -> std::future<std::invoke_result_t<Fn, Args...>>
    {
        using ReturnT = std::invoke_result_t<Fn, Args...>;

        auto task = std::make_shared<std::packaged_task<ReturnT()>>(
            std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...));

        std::future<ReturnT> result = task->get_future();
        {
            std::scoped_lock lock(mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::condition_variable cv_;
    std::mutex mutex_;
    std::atomic<bool> stopping_{ false };
};
