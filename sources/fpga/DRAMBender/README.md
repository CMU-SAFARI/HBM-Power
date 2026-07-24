# Provided DRAM Bender subset (single-chip HBM2-power AE)

This directory contains only the DRAM Bender pieces needed by this artifact's FPGA path: the U55C/HBM2 hardware design sources, the two measurement apps, the API they build against, and the prebuilt bitstreams used by the AE driver. It is not a full upstream DRAM Bender checkout; upstream multi-board installation instructions and example apps are intentionally out of scope here.

## Bitstreams (`prebuilt/XCU55/`)

| Bitstream | Used for |
|---|---|
| `XCU55_no_hbm.bit` | no-HBM idle-offset baseline (all figures) |
| `XCU55_latest_600MHz_chip0.bit` | IDD + structural tests, stack 0 / channels 0–7 (Figs 2–13, 15) |
| `XCU55_latest_600MHz_chip1.bit` | same, stack 1 / channels 8–15 |
| `bram_tracer_chip0.bit` | trace replay, stack 0 (Fig 14 / Table 3) |
| `bram_tracer_chip1.bit` | trace replay, stack 1 |

`../../../scripts/fpga/reprogram_fpga.sh <name>` programs one of these on the
FPGA host (from this directory in the remote artifact checkout), reboots the
host, and re-inits the SoftMC infrastructure.

## Building the apps (on the FPGA host)

```bash
cd sources/apps/Power_structural_variation && make    # -> ./SoftMC_rdwr
cd sources/apps/HBMTraceRunnerBRAM        && make     # -> ./HBMTraceRunnerBRAM
```

Plain g++ (C++11), no external dependencies; each Makefile compiles the app
plus `sources/api/*`. The AE driver (`scripts/fpga/run_fpga_ae.sh`) does this
automatically before each measurement (`BUILD=make`).

## Rebuilding the bitstreams

`projects/U55-HBM/` + `sources/hdl/` are the design sources (Vivado 2020.2).
The per-bitstream build configuration (what distinguishes the `chip0`/`chip1`
and `no_hbm` variants, e.g. channel selection and HBM IP settings) is not
documented here — reprogramming with the shipped `.bit` files is the supported
AE path. If you intend to synthesize new bitstreams or adapt DRAM Bender to your
own board, contact the authors via HotCRP so we can provide setup-specific
guidance.

## DRAM Bender background

DRAM Bender is an experimental FPGA-based memory controller design that can be used to develop tests for DDR4 [SO/R/U]DIMMs and HBM2 chips.

DRAM Bender is the next version of the [SoftMC memory testing infrastructure](https://github.com/CMU-SAFARI/SoftMC). DRAM Bender introduces general purpose registers and a brand new DRAM Bender ISA to provide a programmable memory controller.

## Cite DRAM Bender

Please cite the following paper if you find DRAM Bender useful:

[A. Olgun, H. Hassan, A. G. Yaglikci, Y. C. Tugrul, L. Orosa, H. Luo, M. Patel, O. Ergin, O. Mutlu, "DRAM Bender: An Extensible and Versatile FPGA-based Infrastructure to Easily Test State-of-the-art DRAM Chips", IEEE TCAD, June 2023.](https://ieeexplore.ieee.org/document/10141996)

Link to the PDF: https://arxiv.org/pdf/2211.05838.pdf  

Below is bibtex format for citation.
```
@article{olgun2023drambender,
      title={{DRAM Bender: An Extensible and Versatile FPGA-based Infrastructure to Easily Test State-of-the-art DRAM Chips}}, 
      author={Olgun, Ataberk and Hassan, Hasan and Yaglikci, A. Giray and Tugrul, Yahya Can and Orosa, Lois and Luo, Haocong and Patel, Minesh and Ergin, Oguz and Mutlu, Onur},
      year={2023},
      journal={IEEE TCAD}
}
```

## More DRAM Bender Resources

[DRAM Bender Tutorial Video](https://www.youtube.com/watch?v=FklVEsfdZCI): 43-minute tutorial on how to set up and use DRAM Bender.

[ETH Projects & Seminars Course on DRAM Bender](https://safari.ethz.ch/projects_and_seminars/spring2023/doku.php?id=softmc)

## DRAM Bender Design

This part covers some details regarding DRAM Bender's design and could be helpful for people who want to modify DRAM Bender.

### ISA

DRAM Bender implements a very simple ISA that consists of some DRAM commands, arithmetic operations and control flow instructions. A list of instructions and their encodings can be found [here](https://docs.google.com/spreadsheets/d/18mPiKa1HBoO0OmzAbWRvo5OnIguL6A9tLEIwNeVfOe8/edit?usp=sharing).

DRAM Bender definition proposes 64 bit wide instructions. An instruction can be decoded as an *EXE OP* or a packet of four *DDR OP*s, i.e. each arithmetic or control flow instruction is defined to be 64 bits wide, while a DDR operation is 16 bits wide.

### Hardware Design

DRAM Bender consists of the *frontend* module which facilitates the communication between the host machine and the *pipeline*. DRAM Bender implements a four stage *pipeline*, in which instructions are carried through *fetch*, *decode*, *exe-1* and *exe-2* stages. When encountered, DDR commands will generate command signals at *exe-2* stage which will be decoded in the following cycle by the *ddr4 adapter* and sent to PHY as valid DDR4 commands.

#### Frontend

Frontend module contains the instruction memory of DRAM Bender. The instruction memory is 16 KBs large and can hold 2048 instructions. Frontend encapsulates the maintenance controller which is responsible for issuing periodic DDR commands to the connected DRAM module.

While DRAM Bender is idle (not running a user program) instructions held in maintenance controller's memory will be fetched by the pipeline until Frontend receives a user program from XDMA. Then the program controller will reset and DRAM Bender will start executing the user program.

#### DDR4 Adapter

DRAM Bender emitted DDR signals will be decoded by the DDR4 Adapter and proper (as defined in DDR4 SDRAM IP's specification) DDR4 signals will be communicated to the PHY interface.

Users who wish to use DRAM Bender within designs which communicate to DRAM modules differently (e.g. another Xilinx IP) may want to modify this module.

### Implementing New Instructions

We explain the effort required to extend the system’s functionality over the example of adding the load/store (LD/ST) instructions to DRAM Bender’s ISA’s earlier version, which consisted of only the arithmetic, control, and miscellaneous instruction types listed in II. Adding the LD/ST instructions consists of two main steps:

#### Modifying the hardware design

First, we extend the hardware description of the decode stage to decode the LD/ST instructions into μ-ops. 

```verilog
// decode_stage.v, lines 159-178
else if(instr[`MEM_OFFSET]) begin
  case(instr[`FU_CODE_OFFSET +: 8])
    `LD: begin
      exe_uop_ns[`RS1 +: 4]    = instr[`DEC_RS1 +: 4];
      exe_uop_ns[`RT  +: 4]    = instr[`DEC_RT +: 4];
      exe_uop_ns[`IMD +: 16]   = instr[`DEC_IMD1  +: 16];
      exe_uop_ns[`IS_MEM]      = `HIGH;
      exe_uop_ns[`IS_LD]       = `HIGH;
      exe_uop_ns[`HAS_IMD]     = `HIGH;
    end
    `ST: begin
      exe_uop_ns[`RS1 +: 4]    = instr[`DEC_RS1 +: 4]; // The address base to write to
      exe_uop_ns[`RS2 +: 4]    = instr[`DEC_RT  +: 4]; // The value to write
      exe_uop_ns[`IMD +: 16]   = instr[`DEC_IMD1  +: 16];
      exe_uop_ns[`IS_MEM]      = `HIGH;
      exe_uop_ns[`IS_ST]       = `HIGH;
      exe_uop_ns[`HAS_IMD]     = `HIGH;
    end        
  endcase
end
```

By our design specification, LD/ST instructions access an on-chip memory. Second, we add a scratchpad on-chip memory to DRAM Bender’s design by instantiating a block ram (BRAM) using Vivado’s IP catalog. 

```verilog
// execute_stage.v, lines 113-117
wire              mem_wen;
wire              mem_ren;
wire[9:0]         mem_addr;
wire[31:0]        mem_wdata;
wire[31:0]        mem_rdata;

// execute_stage.v, lines 159-166
scratchpad data_mem(
    .addra(mem_addr),
    .clka(clk),
    .dina(mem_wdata),
    .douta(mem_rdata),
    .ena(mem_wen || mem_ren),
    .wea(mem_wen)
);
```

Third, we modify the hardware description of the execute stage to load data from and store data to the scratchpad memory when load and store instructions are executed. 

```verilog
// exe_pipeline.v, lines 136-142
if(s2_uop[`IS_LD]) begin
  s2_mem_ren = `HIGH;
end
if(s2_uop[`IS_ST]) begin
  s2_mem_wdata = s2_rs2_data;
  s2_mem_wen   = `HIGH;
end
```

#### Modifying the software API

We add two new functions in `instruction.cpp/h`, allowing users to insert load and store instructions into their programs.

```c++
// instruction.cpp, lines 165-188
Inst SMC_LD(int rb, int offset, int rt)
{
  Inst op_code = (uint64_t)0x1 << __IS_MEM;
  Inst fu_code = (uint64_t)__LD << __FU_CODE;
  Inst s_reg   = rb;
  Inst imd1    = offset << __IMD1;
  Inst t_reg   = rt << __RT;

  Inst inst    = op_code | fu_code | s_reg | imd1 | t_reg;

  return inst;
}
Inst SMC_ST(int rb, int offset, int rv)
{
  Inst op_code = (uint64_t)0x1 << __IS_MEM;
  Inst fu_code = (uint64_t)__ST << __FU_CODE;
  Inst b_reg   = rb;
  Inst imd1    = offset << __IMD1;
  Inst v_reg   = rv << __RT; // We cannot have imd1 and rs2 present simultaneously

  Inst inst    = op_code | fu_code | b_reg | imd1 | v_reg;

  return inst;
}
```


## Known Issues:
- Multi Rank SODIMMs are currently not supported.
- Discrepancies between the API in the repository and the API described in our publication: We are working on developing a more clean API as described in the paper, the new API is still work in progress.

You are welcome to contribute to the project. If you find/solve any issues
or port DRAM Bender to a new FPGA board, please contact the people below.

## Contacts:
Ataberk Olgun (ataberk.olgun [at] safari [dot] ethz [dot] ch)  
Hasan Hassan (hasan.hasan [at] safari [dot] ethz [dot] ch)  
