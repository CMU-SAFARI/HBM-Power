#include "board.h"
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <cassert>
#include <fcntl.h>
#include <string.h>
#include <cstdint>   // uint64_t (required on newer GCC/libstdc++ that don't pull it in transitively)

BoardInterface::BoardInterface(IFACE iface_type, int dimm_select, bool garbage_reads_enabled)
{
  this -> iface_type = iface_type;
  this -> dimm_select = dimm_select;
  this -> garbage_reads_enabled = garbage_reads_enabled;
}
BoardInterface::~BoardInterface()
{
  free(send_buf);
  free(recv_buf);
  close(to_card);
  close(from_card);
}

void BoardInterface::closeReceiveDescriptor()
{
  if (from_card >= 0) {
    close(from_card);
    from_card = -1;
    std::cout << "Closed receive descriptor" << std::endl;
  }
}

int BoardInterface::init()
{
  switch(iface_type)
  {
    case IFACE::XDMA:
    {
      int fpga_fd;
      std::string TO_FPGA_OPENED = "NULL";
      std::string FROM_FPGA_OPENED = "NULL";
  
  
      if (dimm_select == 0){
        fpga_fd = open(TO_FPGA_DEFAULT.c_str(), O_RDWR);
	TO_FPGA_OPENED = TO_FPGA_DEFAULT ; 
	}
     
      else if (dimm_select == 1)
     	{
	      fpga_fd = open("/dev/xdma1_h2c_0", O_RDWR);
	      TO_FPGA_OPENED = "/dev/xdma1_h2c_0" ; 
      	}
      
      else if (dimm_select == 2)
     	{
	      fpga_fd = open("/dev/xdma2_h2c_0", O_RDWR);
	      TO_FPGA_OPENED = "/dev/xdma2_h2c_0" ; 
      	}
      
      else if (dimm_select == 3)
     	{
	      fpga_fd = open("/dev/xdma3_h2c_0", O_RDWR);
	      TO_FPGA_OPENED = "/dev/xdma3_h2c_0" ; 
      	}
      
      else if (dimm_select == 4)
     	{
	      fpga_fd = open("/dev/xdma4_h2c_0", O_RDWR);
	      TO_FPGA_OPENED = "/dev/xdma4_h2c_0" ; 
      	}
      
      else if (dimm_select == 5)
     	{
	      fpga_fd = open("/dev/xdma5_h2c_0", O_RDWR);
	      TO_FPGA_OPENED = "/dev/xdma5_h2c_0" ; 
      	}

      else if (dimm_select == 6)
     	{
	      fpga_fd = open("/dev/xdma6_h2c_0", O_RDWR);
	      TO_FPGA_OPENED = "/dev/xdma6_h2c_0" ; 
      	}
  
      else if (dimm_select == 7)
     	{
	      fpga_fd = open("/dev/xdma7_h2c_0", O_RDWR);
	      TO_FPGA_OPENED = "/dev/xdma7_h2c_0" ; 
      	}

      if(fpga_fd<0)
      {
        std::cerr << "Open to card failed!" << std::endl;
        return 1;
      }
      else
        //std::cout << "Opened " << (dimm_select == 0 ? TO_FPGA_DEFAULT : "/dev/xdma0_h2c_1") <<  " -> " << fpga_fd << std::endl;
        std::cout << "Opened " << TO_FPGA_OPENED  <<  " -> " << fpga_fd << std::endl;
     
        to_card = fpga_fd;
  
      if (dimm_select == 0) {
        fpga_fd = open(FROM_FPGA_DEFAULT.c_str(), O_RDWR);
   	FROM_FPGA_OPENED = FROM_FPGA_DEFAULT ;
      }
      else if (dimm_select == 1) {
        fpga_fd = open("/dev/xdma1_c2h_0", O_RDWR | O_TRUNC);
   	FROM_FPGA_OPENED = "/dev/xdma1_c2h_0" ;
 	}
 
      else if (dimm_select == 2) {
        fpga_fd = open("/dev/xdma2_c2h_0", O_RDWR | O_TRUNC);
   	FROM_FPGA_OPENED = "/dev/xdma2_c2h_0" ;
 	}
 
      else if (dimm_select == 3) {
        fpga_fd = open("/dev/xdma3_c2h_0", O_RDWR | O_TRUNC);
   	FROM_FPGA_OPENED = "/dev/xdma3_c2h_0" ;
 	}
 
      else if (dimm_select == 4) {
        fpga_fd = open("/dev/xdma4_c2h_0", O_RDWR | O_TRUNC);
   	FROM_FPGA_OPENED = "/dev/xdma4_c2h_0" ;
 	}
 
      else if (dimm_select == 5) {
        fpga_fd = open("/dev/xdma5_c2h_0", O_RDWR | O_TRUNC);
   	FROM_FPGA_OPENED = "/dev/xdma5_c2h_0" ;
 	}
    
      else if (dimm_select == 6) {
        fpga_fd = open("/dev/xdma6_c2h_0", O_RDWR | O_TRUNC);
   	FROM_FPGA_OPENED = "/dev/xdma6_c2h_0" ;
 	}
    
      else if (dimm_select == 7) {
        fpga_fd = open("/dev/xdma7_c2h_0", O_RDWR | O_TRUNC);
   	FROM_FPGA_OPENED = "/dev/xdma7_c2h_0" ;
 	}
    
    
      if(fpga_fd<0)
      {
        std::cerr << "Open to host failed!" << std::endl;
        return 1;
      }
      else
        //std::cout << "Opened " << (dimm_select == 0 ? FROM_FPGA_DEFAULT : "/dev/xdma0_c2h_1") <<  " -> " << fpga_fd << std::endl;
        std::cout << "Opened " << FROM_FPGA_OPENED  <<  " -> " << fpga_fd << std::endl;
     
     
      from_card = fpga_fd;
      // allocate page size aligned X page size regions to our buffers
      if (posix_memalign((void **)&send_buf, 4096 /*alignment */ , SEND_BUF_SIZE + (4096-(SEND_BUF_SIZE % 4096))) != 0)
      {
        std::cerr << "Send buffer allocation failed!" << std::endl;
        return 1;
      }
      if (posix_memalign((void **)&recv_buf, 4096 /*alignment */ , RECV_BUF_SIZE + (4096-(RECV_BUF_SIZE % 4096))))
      {
        std::cerr << "Receive buffer allocation failed!" << std::endl;
        return 1;
      }
      if( (!send_buf) || (!recv_buf) )
      {
        std::cerr << "Buffers cannot be allocated!" << std::endl;
        return 1;
      }
      return 0;
    }
    default:
      std::cerr << "Unknown iface_type!" << std::endl;
      return 1;
  }
}

int BoardInterface::sendData(void* data, const uint size)
{
  switch(iface_type)
  {
    case IFACE::XDMA:
      return xdma_send(data,size);
      break;
    default:
      std::cerr << "Unknown iface_type!" << std::endl;
      return 1;
  }
}

int BoardInterface::recvData(void* buf, const uint size)
{
  switch(iface_type)
  {
    case IFACE::XDMA:
      return xdma_recv(buf,size);
      break;
    default:
      std::cerr << "Unknown iface_type!" << std::endl;
      return 1;
  }
}

int BoardInterface::xdma_send(void* data, const uint size)
{
  memcpy((char*)send_buf, (char*)data, size);

  int fd = to_card;
  ssize_t rc;
  uint64_t count = 0;
  char *buf = (char*) send_buf;
 
  while (count < size) {
    /* write data to file from memory buffer */
    rc = write(fd, buf, size);
    // std::cout << rc << std::endl;
    assert(rc == size || rc == 0);
    count += rc;
  }

  // We wrote more than we supposed to
  if (count != size)
    return 1;

  return 0;
}

int BoardInterface::xdma_send_direct(const void* data, uint size)
{
  int fd = to_card;
  ssize_t rc;
  uint64_t count = 0;
  const char *buf = (const char*) data;

  while (count < size) {
    rc = write(fd, buf + count, size - count);
    if (rc < 0) return 1;
    count += rc;
  }

  return 0;
}

int BoardInterface::sendDataDirect(const void* data, uint size)
{
  switch(iface_type)
  {
    case IFACE::XDMA:
      return xdma_send_direct(data, size);
    default:
      std::cerr << "Unknown iface_type!" << std::endl;
      return 1;
  }
}

int BoardInterface::xdma_recv(void* buf, const uint size)
{
  assert(size <= RECV_BUF_SIZE && "given read size is too large");

  int fd = from_card;
  int64_t count = 0;
  // Try to read from the card.
  if (garbage_reads_enabled)
  {
    while (garbage_reads_enabled && from_card >= 0)
    {
      count = read(from_card, (char*) recv_buf, size);
      // std::cout << "Garbage read size: " << count << std::endl;
      if (count != -1)
        break;
      // std::cout << "Garbage read retrying..." << std::endl;
    }

    std::cout << "Exiting garbage_reads loop" << std::endl;
  }
  else
  {
    count = read(fd, (char*) recv_buf, size);
  }

  // // print errno
  // // if (count < 0) {
  //   std::cerr << "Read error, errno: " << errno << std::endl;
  //   // print errstr
  //   std::cerr << "Read error, errstr: " << strerror(errno) << std::endl;
  //   // return -1;
  // // }

  // // print size, count
  // std::cout << "Requested size: " << size << ", Read size: " << count << std::endl;
  assert(count <= size);// || rc == 0);

  memcpy(buf, (char*) recv_buf, count);

  // We read more than we were supposed to
  return count;
}
