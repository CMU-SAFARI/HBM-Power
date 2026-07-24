#include "DRAMPower/standards/hbm3/HBM3.h"

#include <algorithm>
#include <iostream>

#include "DRAMPower/standards/hbm3/core_calculation_HBM3.h"

namespace DRAMPower {

    HBM3::HBM3(const MemSpecHBM3 &memSpec)
        : dram_base<CmdType>(PatternEncoderOverrides{})
        , memSpec(memSpec)
        , ranks(1, {(std::size_t)memSpec.banksPerPseudoChannel})
    {
        this->registerCommands();
    }

    void HBM3::registerCommands() {
        // ACT
        this->registerBankHandler<CmdType::ACT>(&HBM3::handleAct);
        // PRE
        this->registerBankHandler<CmdType::PRE>(&HBM3::handlePre);
        // PREA
        this->registerRankHandler<CmdType::PREA>(&HBM3::handlePreAll);
        // RD
        this->registerBankHandler<CmdType::RD>(&HBM3::handleRead);
        // WR
        this->registerBankHandler<CmdType::WR>(&HBM3::handleWrite);
        // RDA
        this->registerBankHandler<CmdType::RDA>(&HBM3::handleReadAuto);
        // WRA
        this->registerBankHandler<CmdType::WRA>(&HBM3::handleWriteAuto);
        // EOS
        routeCommand<CmdType::END_OF_SIMULATION>([this](const Command &cmd) {
            this->endOfSimulation(cmd.timestamp);
        });
    }

// Getters
    uint64_t HBM3::getBankCount() {
        return memSpec.banksPerPseudoChannel;
    }

    uint64_t HBM3::getRankCount() {
        return 1; // pseudo-channel modeled as single rank
    }

    uint64_t HBM3::getDeviceCount() {
        return 1;
    }

// Command handlers
    void HBM3::handleAct(Rank &rank, Bank &bank, timestamp_t timestamp) {
        bank.counter.act++;
        bank.cycles.act.start_interval(timestamp);

        if (!rank.isActive(timestamp)) {
            rank.cycles.act.start_interval(timestamp);
        }

        bank.bankState = Bank::BankState::BANK_ACTIVE;
    }

    void HBM3::handlePre(Rank &rank, Bank &bank, timestamp_t timestamp) {
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

    void HBM3::handlePreAll(Rank &rank, timestamp_t timestamp) {
        for (auto &bank : rank.banks) {
            handlePre(rank, bank, timestamp);
        }
    }

    void HBM3::handleRead(Rank &, Bank &bank, timestamp_t) {
        ++bank.counter.reads;
    }

    void HBM3::handleWrite(Rank &, Bank &bank, timestamp_t) {
        ++bank.counter.writes;
    }

    void HBM3::handleReadAuto(Rank &rank, Bank &bank, timestamp_t timestamp) {
        ++bank.counter.readAuto;

        auto minBankActiveTime = bank.cycles.act.get_start() + memSpec.memTimingSpec.tRAS;
        auto minReadActiveTime = timestamp + memSpec.prechargeOffsetRD;
        auto delayed_timestamp = std::max(minBankActiveTime, minReadActiveTime);

        addImplicitCommand(delayed_timestamp, [this, &rank, &bank, delayed_timestamp]() {
            this->handlePre(rank, bank, delayed_timestamp);
        });
    }

    void HBM3::handleWriteAuto(Rank &rank, Bank &bank, timestamp_t timestamp) {
        ++bank.counter.writeAuto;

        auto minBankActiveTime = bank.cycles.act.get_start() + memSpec.memTimingSpec.tRAS;
        auto minWriteActiveTime = timestamp + memSpec.prechargeOffsetWR;
        auto delayed_timestamp = std::max(minBankActiveTime, minWriteActiveTime);

        addImplicitCommand(delayed_timestamp, [this, &rank, &bank, delayed_timestamp]() {
            this->handlePre(rank, bank, delayed_timestamp);
        });
    }

    void HBM3::endOfSimulation(timestamp_t) {
        if (this->implicitCommandCount() > 0)
            std::cout << "[WARN] End of simulation but still implicit commands left!" << std::endl;
    }

// Calculation
    energy_t HBM3::calcCoreEnergy(timestamp_t timestamp) {
        Calculation_HBM3 calculation;
        return calculation.calcEnergy(timestamp, *this);
    }

    interface_energy_info_t HBM3::calcInterfaceEnergy(timestamp_t) {
        // No interface power modeled for HBM3
        return interface_energy_info_t{};
    }

// Stats
    SimulationStats HBM3::getWindowStats(timestamp_t timestamp) {
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

    SimulationStats HBM3::getStats() {
        return getWindowStats(getLastCommandTime());
    }

} // namespace DRAMPower
