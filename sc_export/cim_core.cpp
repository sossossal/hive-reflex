/**
 * Hive-Reflex CIM Core - Implementation
 */

#include "cim_core.h"

void CIM_Core::b_transport(tlm::tlm_generic_payload& trans, sc_time& delay) {
    tlm::tlm_command cmd = trans.get_command();
    uint64_t addr = trans.get_address();
    unsigned char* ptr = trans.get_data_ptr();
    unsigned int len = trans.get_data_length();

    if (cmd == tlm::TLM_WRITE_COMMAND) {
        // Check for ISA specific addresses or Registers
        if (addr == 0x4000) {
             // Command Register
             REG_CTRL = *((uint32_t*)ptr);
             // Trigger Compute Event (not shown)
        }
    } else if (cmd == tlm::TLM_READ_COMMAND) {
        // Read Status
        if (addr == 0x4004) {
             *((uint32_t*)ptr) = REG_STATUS;
        }
    }
    
    // Simulate Latency
    delay += sc_time(10, SC_NS);
    
    trans.set_response_status(tlm::TLM_OK_RESPONSE);
}

void CIM_Core::compute_process() {
    while(true) {
        wait(10, SC_NS); 
        // Compute logic would go here
    }
}
