#include "prefix_cache.h"
#include "request_scheduler.h"

#include <iostream>
#include <vector>

int main() {
    PrefixCacheManager cache(128);
    RequestScheduler scheduler(&cache, 4);

    Request req1;
    req1.request_id = 1;
    req1.tokens = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    Request req2;
    req2.request_id = 2;
    req2.tokens = {10, 11, 12, 13, 14, 15, 16, 17};

    scheduler.addRequest(req1);
    scheduler.addRequest(req2);

    for (int step = 0; step < 5; ++step) {
        auto results = scheduler.step();

        std::cout << "step=" << step
                  << " waiting=" << scheduler.waitingCount()
                  << " running=" << scheduler.runningCount()
                  << " finished=" << scheduler.finishedCount()
                  << "\n";

        for (const auto& r : results) {
            std::cout << "  req=" << r.request_id
                      << " block=" << r.block_id
                      << " prefill_cursor=" << r.prefill_cursor
                      << " decode_len=" << r.decode_len
                      << " state=" << static_cast<int>(r.state)
                      << "\n";
        }
    }

    return 0;
}