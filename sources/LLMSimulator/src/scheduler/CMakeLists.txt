add_library(scheduler OBJECT)

target_sources(
    scheduler PRIVATE

    scheduler.cpp
    sequence.cpp
)