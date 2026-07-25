#pragma once

#include "prefix_cache.h"

#include <cstddef>
#include <vector>

enum class RequestState {
    Waiting,
    Prefill,
    Decoding,
    Finished
};

struct Request {
    int request_id = -1;
    std::vector<int> tokens;
    std::vector<int> block_ids;
    RequestState state = RequestState::Waiting;
    size_t prefill_cursor = 0;
    size_t decode_len = 0;
    int slot = -1;
};

struct StepResult {
    int request_id = -1;
    int block_id = -1;
    size_t prefill_cursor = 0;
    size_t decode_len = 0;
    RequestState state = RequestState::Waiting;
};

class RequestScheduler {
public:
    RequestScheduler(PrefixCacheManager* cache, size_t chunk_size = 16);

    void addRequest(const Request& request);
    Request* startNextWaitingRequest(int slot);
    Request* getRequestById(int request_id);
    bool finalizeRunningRequest(int request_id);
    std::vector<StepResult> step();

    size_t waitingCount() const;
    size_t runningCount() const;
    size_t finishedCount() const;

private:
    PrefixCacheManager* cache_ = nullptr;
    size_t chunk_size_ = 16;
    std::vector<Request> waiting_queue_;
    std::vector<Request> running_queue_;
    std::vector<Request> finished_queue_;
};