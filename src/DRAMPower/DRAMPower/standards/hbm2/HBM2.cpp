#include "DRAMPower/standards/hbm2/HBM2.h"

#include <algorithm>
#include <iostream>

#include "DRAMPower/standards/hbm2/core_calculation_HBM2.h"

namespace DRAMPower {

    HBM2::HBM2(const MemSpecHBM2 &memSpec)
        : dram_base<CmdType>(PatternEncoderOverrides{})
        , memSpec(memSpec)
        , ranks(1, {(std::size_t)memSpec.banksPerPseudoChannel})
    {
        this->registerCommands();
    }

    void HBM2::registerCommands() {
        // ACT
        this->registerBankHandler<CmdType::ACT>(&HBM2::handleAct);
        // PRE
        this->registerBankHandler<CmdType::PRE>(&HBM2::handlePre);
        // PREA
        this->registerRankHandler<CmdType::PREA>(&HBM2::handlePreAll);
        this->registerRankHandler<CmdType::REFA>(&HBM2::handleRefAll);
        // RD
        this->registerBankHandler<CmdType::RD>(&HBM2::handleRead);
        // WR
        this->registerBankHandler<CmdType::WR>(&HBM2::handleWrite);
        // RDA
        this->registerBankHandler<CmdType::RDA>(&HBM2::handleReadAuto);
        // WRA
        this->registerBankHandler<CmdType::WRA>(&HBM2::handleWriteAuto);
        // EOS
        routeCommand<CmdType::END_OF_SIMULATION>([this](const Command &cmd) {
            this->endOfSimulation(cmd.timestamp);
        });
    }

// Getters
    uint64_t HBM2::getBankCount() {
        return memSpec.banksPerPseudoChannel;
    }

    uint64_t HBM2::getRankCount() {
        return 1; // pseudo-channel modeled as single rank
    }

    uint64_t HBM2::getDeviceCount() {
        return 1;
    }

// Command handlers
    void HBM2::handleAct(Rank &rank, Bank &bank, timestamp_t timestamp) {
        bank.counter.act++;
        bank.cycles.act.start_interval(timestamp);

        if (!rank.isActive(timestamp)) {
            rank.cycles.act.start_interval(timestamp);
        }

        bank.bankState = Bank::BankState::BANK_ACTIVE;
    }

    void HBM2::handlePre(Rank &rank, Bank &bank, timestamp_t timestamp) {
        if (bank.bankState == Bank::BankState::BANK_PRECHARGED)
            return;

        bank.counter.pre++;
        bank.cycles.act.close_interval(timestamp);
        bank.latestPre = timestamp;
        bank.bankState = Bank::BankState::BANK_PRECHARGED;

        if (!rank.isActive(timestamp)) {
            rank.cycles.act.close_interval(timestamp);
        }
    }

    void HBM2::handlePreAll(Rank &rank, timestamp_t timestamp) {
        for (auto &bank : rank.banks) {
            handlePre(rank, bank, timestamp);
        }
    }

    // All-bank refresh: every bank is busy (active-like) for tRFC, then implicitly
    // precharged. Same as HBM3::handleRefAll (and DDR5::handleRefAll).
    void HBM2::handleRefAll(Rank &rank, timestamp_t timestamp) {
        for (auto &bank : rank.banks) {
            handleRefreshOnBank(rank, bank, timestamp, memSpec.memTimingSpec.tRFC, bank.counter.refAllBank);
        }
        rank.endRefreshTime = timestamp + memSpec.memTimingSpec.tRFC;
    }

    void HBM2::handleRefreshOnBank(Rank &rank, Bank &bank, timestamp_t timestamp, uint64_t timing, uint64_t &counter) {
        ++counter;
        if (!rank.isActive(timestamp)) {
            rank.cycles.act.start_interval(timestamp);
        }
        bank.bankState = Bank::BankState::BANK_ACTIVE;
        auto timestamp_end = timestamp + timing;
        bank.refreshEndTime = timestamp_end;
        if (!bank.cycles.act.is_open())
            bank.cycles.act.start_interval(timestamp);
        addImplicitCommand(timestamp_end, [this, &bank, &rank, timestamp_end]() {
            bank.bankState = Bank::BankState::BANK_PRECHARGED;
            bank.cycles.act.close_interval(timestamp_end);
            if (!rank.isActive(timestamp_end)) {
                rank.cycles.act.close_interval(timestamp_end);
            }
        });
    }

    void HBM2::handleRead(Rank &, Bank &bank, timestamp_t) {
        ++bank.counter.reads;
    }

    void HBM2::handleWrite(Rank &, Bank &bank, timestamp_t) {
        ++bank.counter.writes;
    }

    void HBM2::handleReadAuto(Rank &rank, Bank &bank, timestamp_t timestamp) {
        ++bank.counter.readAuto;

        auto minBankActiveTime = bank.cycles.act.get_start() + memSpec.memTimingSpec.tRAS;
        auto minReadActiveTime = timestamp + memSpec.prechargeOffsetRD;
        auto delayed_timestamp = std::max(minBankActiveTime, minReadActiveTime);

        addImplicitCommand(delayed_timestamp, [this, &rank, &bank, delayed_timestamp]() {
            this->handlePre(rank, bank, delayed_timestamp);
        });
    }

    void HBM2::handleWriteAuto(Rank &rank, Bank &bank, timestamp_t timestamp) {
        ++bank.counter.writeAuto;

        auto minBankActiveTime = bank.cycles.act.get_start() + memSpec.memTimingSpec.tRAS;
        auto minWriteActiveTime = timestamp + memSpec.prechargeOffsetWR;
        auto delayed_timestamp = std::max(minBankActiveTime, minWriteActiveTime);

        addImplicitCommand(delayed_timestamp, [this, &rank, &bank, delayed_timestamp]() {
            this->handlePre(rank, bank, delayed_timestamp);
        });
    }

    void HBM2::endOfSimulation(timestamp_t) {
        if (this->implicitCommandCount() > 0)
            std::cout << "[WARN] End of simulation but still implicit commands left!" << std::endl;
    }

// Calculation
    energy_t HBM2::calcCoreEnergy(timestamp_t timestamp) {
        Calculation_HBM2 calculation;
        return calculation.calcEnergy(timestamp, *this);
    }

    interface_energy_info_t HBM2::calcInterfaceEnergy(timestamp_t) {
        // No interface power modeled for HBM2
        return interface_energy_info_t{};
    }

// Stats
    SimulationStats HBM2::getWindowStats(timestamp_t timestamp) {
        processImplicitCommandQueue(timestamp);

        SimulationStats stats;
        auto B = memSpec.banksPerPseudoChannel;
        stats.bank.resize(B);
        stats.rank_total.resize(1);

        auto simulation_duration = timestamp;
        Rank &rank = ranks[0];

        for (std::size_t b = 0; b < B; ++b) {
            stats.bank[b].counter = rank.banks[b].counter;
            stats.bank[b].cycles.act = rank.banks[b].cycles.act.get_count_at(timestamp);
            stats.bank[b].cycles.ref = rank.banks[b].cycles.ref.get_count_at(timestamp);
            stats.bank[b].cycles.pre =
                simulation_duration - stats.bank[b].cycles.act;
        }

        stats.rank_total[0].cycles.act = rank.cycles.act.get_count_at(timestamp);
        stats.rank_total[0].cycles.pre =
            simulation_duration - rank.cycles.act.get_count_at(timestamp);

        return stats;
    }

    SimulationStats HBM2::getStats() {
        return getWindowStats(getLastCommandTime());
    }

} // namespace DRAMPower
