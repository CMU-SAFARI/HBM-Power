#include <string>

#ifndef BOARD_H
#define BOARD_H
/** This class defines how the host
 * interfaces with the board.
 */
class BoardInterface{
  const uint SEND_BUF_SIZE = 32*2048*64;
              // send each instruction as a 256-bit packet
  const uint RECV_BUF_SIZE = 1024*128;
  const std::string TO_FPGA_DEFAULT = "/dev/xdma0_h2c_0";
  const std::string FROM_FPGA_DEFAULT = "/dev/xdma0_c2h_0";
public:
  enum class IFACE {
      XDMA = 0
  };
  BoardInterface(IFACE, int dimm_select = 0, bool garbage_reads_enabled = false);
  ~BoardInterface();
  int init();
  int sendData(void* data, const uint size);
  int sendDataDirect(const void* data, uint size);
  int recvData(void* buf , const uint size);

  /**
   * Disable garbage reads, allowing the xdma_recv loop to exit
   */
  void disableGarbageReads() { garbage_reads_enabled = false; }

  /**
   * Close the receive file descriptor to interrupt any blocked read() calls
   */
  void closeReceiveDescriptor();
private:
  IFACE iface_type;
  // XDMA related constructs
  int to_card;
  int from_card;
  int dimm_select;
  void* send_buf;
  void* recv_buf;
  bool garbage_reads_enabled = false;
  int xdma_send(void* data, const uint size);
  int xdma_send_direct(const void* data, uint size);
  int xdma_recv(void* buf,  const uint size);
  // end XDMA related constructs
};

#endif
