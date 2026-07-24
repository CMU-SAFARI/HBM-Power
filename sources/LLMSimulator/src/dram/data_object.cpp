#include "dram/data_object.h"

#include <memory>
#include <vector>

#include "common/assert.h"
#include "common/type.h"

namespace llm_system {

DataObject::DataObject(MMap mmap) : mmap(mmap){};
DataObject::DataObject(const Address &addr) : addr_{addr} {};

const int DataObject::GetRowAddr(int offset) {
  assertTrue(offset < addr_.num_chunks_, "not valid offset");
  int row_stride = offset / addr_.num_column_groups_;
  return addr_.row_start_index_ + row_stride;
}

const int DataObject::GetColAddr(int offset) {
  assertTrue(offset < addr_.num_chunks_, "not valid offset");
  int column_stride = offset % addr_.num_column_groups_;
  return addr_.column_group_start_index_ + column_stride;
}

}  // namespace llm_system
