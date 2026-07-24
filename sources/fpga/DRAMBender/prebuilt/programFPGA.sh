#!/bin/bash
board_id=0

board=${1}
bitstream=${2}

bitfile_name=${board}/${bitstream}.bit
probesfile_name=${board}/${bitstream}.ltx

if [ -z "$VIVADO_EXEC" ]
then
  echo "Please assign vivado executable's path to VIVADO_EXEC variable first!"
else
  echo "Trying to program the board with the prebuilt files ${bitfile_name}..."
  $VIVADO_EXEC -mode tcl -source $( dirname "${BASH_SOURCE[0]}")/programFPGA.tcl -nolog -nojournal -tclargs ${bitfile_name} ${probesfile_name}
  echo "Done programming the board!"
fi
