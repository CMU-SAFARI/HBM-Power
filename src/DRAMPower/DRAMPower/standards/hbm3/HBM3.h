#ifndef DRAMPOWER_STANDARDS_HBM3_HBM3_H
#define DRAMPOWER_STANDARDS_HBM3_HBM3_H

#include <cstdint>
#include <deque>
#include <vector>
#include <optional>

#include "DRAMPower/Types.h"
#include "DRAMPower/command/Command.h"
#include "DRAMPower/data/energy.h"
#include "DRAMPower/dram/Rank.h"
#include "DRAMPower/dram/dram_base.h"
#include "DRAMPower/memspec/MemSpecHBM3.h"

namespace DRAMPower {

class HBM3 : public dram_base<CmdType> {

public:
    MemSpecHBM3 memSpec;

    // A pseudo-channel is modeled as a single Rank with banksPerPseudoChannel banks
    std::vector<Rank> ranks;

public:
    HBM3(const MemSpecHBM3 &memSpec);
    virtual ~HBM3() = default;

    SimulationStats getStats() override;
    uint64_t getBankCount() override;
    uint64_t getRankCount() override;
    uint64_t getDeviceCount() override;

    energy_t calcCoreEnergy(timestamp_t timestamp) override;
    interface_energy_info_t calcInterfaceEnergy(timestamp_t timestamp) override;

    // Command handlers
    void handleAct(Rank &rank, Bank &bank, timestamp_t timestamp);
    void handlePre(Rank &rank, Bank &bank, timestamp_t timestamp);
    void handlePreAll(Rank &rank, timestamp_t timestamp);
    void handleRefAll(Rank &rank, timestamp_t timestamp);
    void handleRefreshOnBank(Rank &rank, Bank &bank, timestamp_t timestamp, uint64_t timing, uint64_t &counter);
    void handleRead(Rank &rank, Bank &bank, timestamp_t timestamp);
    void handleWrite(Rank &rank, Bank &bank, timestamp_t timestamp);
    void handleReadAuto(Rank &rank, Bank &bank, timestamp_t timestamp);
    void handleWriteAuto(Rank &rank, Bank &bank, timestamp_t timestamp);
    void endOfSimulation(timestamp_t timestamp);

    SimulationStats getWindowStats(timestamp_t timestamp);

protected:
    template <dram_base::commandEnum_t Cmd, typename Func>
    void registerBankHandler(Func &&member_func) {
        this->routeCommand<Cmd>([this, member_func](const Command &command) {
            auto &rank = this->ranks.at(command.targetCoordinate.rank);
            auto &bank = rank.banks.at(command.targetCoordinate.bank);
            rank.commandCounter.inc(command.type);
            (this->*member_func)(rank, bank, command.timestamp);
        });
    }

    template <dram_base::commandEnum_t Cmd, typename Func>
    void registerRankHandler(Func &&member_func) {
        this->routeCommand<Cmd>([this, member_func](const Command &command) {
            auto &rank = this->ranks.at(command.targetCoordinate.rank);
            rank.commandCounter.inc(command.type);
            (this->*member_func)(rank, command.timestamp);
        });
    }

    template <dram_base::commandEnum_t Cmd, typename Func>
    void registerHandler(Func &&member_func) {
        this->routeCommand<Cmd>([this, member_func](const Command &command) {
            (this->*member_func)(command.timestamp);
        });
    }

    void registerCommands();

private:
    uint64_t getInitEncoderPattern() override { return 0; }
    timestamp_t update_toggling_rate(timestamp_t timestamp,
        const std::optional<DRAMUtils::Config::ToggleRateDefinition> &) override {
        return timestamp;
    }
};

} // namespace DRAMPower

#endif /* DRAMPOWER_STANDARDS_HBM3_HBM3_H */
