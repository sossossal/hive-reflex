
/**
 * OpenNeuro Network Controller
 * for Hive-Reflex SoC
 * 
 * Provides basic register interface for PTP stack integration
 * and packet loopback for verification.
 * 
 * Performance Upgrade:
 * - 1KB FIFO for TX/RX
 * - Performance Counters
 */

`timescale 1ns / 1ps

module openneuro_network_ctrl #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    
    // AHB Slave Interface
    input wire [ADDR_WIDTH-1:0] haddr,
    input wire [DATA_WIDTH-1:0] hwdata,
    output reg [DATA_WIDTH-1:0] hrdata,
    input wire hwrite,
    input wire [2:0] hsize,
    input wire [1:0] htrans,
    input wire hsel,
    input wire hready_in, 
    output reg hready_out,
    output reg hresp,
    
    // Interrupt
    output reg irq,
    
    // SMI Interface (MDC/MDIO) for BASE-T1 PHY
    output reg mdc,
    output reg mdio_out,
    output reg mdio_oe,
    input wire mdio_in,
    
    // PTP PPS
    output reg pps_out,
    
    // Reflex Trigger
    output reg reflex_trig
);

    // Register Map
    // ...
    // 0x30: REFLEX_MATCH (32-bit Match Pattern)
    // 0x34: REFLEX_MASK (32-bit Mask)
    // 0x38: REFLEX_CTRL (Bit 0: Enable)
    
    reg [31:0] regs [0:14]; // Expanded regs
    
    // ... (Existing logic)

    // Reflex Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reflex_trig <= 0;
            regs[12] <= 0; // REFLEX_MATCH
            regs[13] <= 0; // REFLEX_MASK
            regs[14] <= 0; // REFLEX_CTRL
        end else begin
             // Pulse generation for trigger
             if (reflex_trig) reflex_trig <= 0;
             
             // Simple Reflex: Check if written data matches pattern
             if (hwrite && hsel && hready_in && htrans[1] && reg_idx == 4'h4) begin // Write to TX (Loopback path) or RX push
                 // In real usage, this checks RX data from PHY. 
                 // Here we check the data being pushed to RX FIFO (Loopback or from PHY logic)
                 // Assuming hwdata is the packet data word.
                 if (regs[14][0]) begin // Enabled
                     if ((hwdata & regs[13]) == (regs[12] & regs[13])) begin
                         reflex_trig <= 1;
                     end
                 end
             end
        end
    end
    
    // FIFOs (1KB = 256 x 32bit)
    reg [31:0] tx_fifo [0:255];
    reg [7:0] tx_wr_ptr, tx_rd_ptr;
    reg [31:0] rx_fifo [0:255];
    reg [7:0] rx_wr_ptr, rx_rd_ptr;
    
    // Performance Logic
    reg [31:0] packet_start_time;
    
    // Internal State
    reg [63:0] ptp_counter;
    
    // SMI State Machine
    localparam SMI_IDLE = 0, SMI_PREAMBLE = 1, SMI_START = 2, SMI_OP = 3, SMI_PHYA = 4, SMI_REGA = 5, SMI_TA = 6, SMI_DATA = 7;
    reg [2:0] smi_state;
    reg [5:0] smi_bit_cnt;
    reg smi_write_op;
    
    // Address Decoding (offset)
    wire [3:0] reg_idx = haddr[5:2];
    
    // PTP Timer & Perf Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ptp_counter <= 0;
            regs[10] <= 0; // PERF_RX_COUNT
            regs[11] <= 0; // PERF_LAST_LATENCY
        end else begin
            ptp_counter <= ptp_counter + 10; 
            
            // Loopback Performance Measurement
            // If data moved from TX to RX (Loopback), measure latency
            if (regs[0][1] && (tx_wr_ptr != tx_rd_ptr)) begin
                // Update stats
            end
            
            // PPS Generation (1Hz 50% duty cycle)
            // ptp_counter is ns. 1s = 10^9 ns.
            // Toggle every 0.5s = 5*10^8 ns.
            if (ptp_counter % 1000000000 < 500000000) begin
                pps_out <= 1;
            end else begin
                pps_out <= 0;
            end
        end
    end
    
    // SMI Logic
    reg [7:0] mdc_div;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mdc <= 0;
            mdc_div <= 0;
            smi_state <= SMI_IDLE;
            regs[9] <= 0; // SMI_STATUS
            mdio_oe <= 0;
            mdio_out <= 0;
        end else begin
            mdc_div <= mdc_div + 1;
            if (mdc_div == 49) begin 
                mdc <= ~mdc;
                if (mdc == 1) begin // Falling edge logic - Simplified for brevity in this update
                    // (Keep existing SMI logic or simplified version)
                     case (smi_state)
                        SMI_IDLE: begin
                            if (regs[6][0]) begin 
                                smi_state <= SMI_PREAMBLE;
                                smi_bit_cnt <= 31;
                                mdio_oe <= 1;
                                mdio_out <= 1;
                                smi_write_op <= regs[6][1];
                                regs[9][0] <= 1;
                                regs[6][0] <= 0;
                            end else begin
                                regs[9][0] <= 0;
                                mdio_oe <= 0;
                            end
                        end
                         SMI_PREAMBLE: begin
                            mdio_out <= 1;
                            if (smi_bit_cnt == 0) begin
                                smi_state <= SMI_START;
                                smi_bit_cnt <= 1;
                            end else smi_bit_cnt <= smi_bit_cnt - 1;
                        end
                        SMI_START: begin
                            mdio_out <= (smi_bit_cnt == 1) ? 0 : 1;
                            if (smi_bit_cnt == 0) begin 
                                smi_state <= SMI_OP;
                                smi_bit_cnt <= 1;
                            end else smi_bit_cnt <= smi_bit_cnt - 1;
                        end
                        SMI_OP: begin
                            mdio_out <= (smi_bit_cnt == 1) ? (smi_write_op ? 0 : 1) : (smi_write_op ? 1 : 0);
                            if (smi_bit_cnt == 0) begin
                                smi_state <= SMI_PHYA;
                                smi_bit_cnt <= 4;
                            end else smi_bit_cnt <= smi_bit_cnt - 1;
                        end
                        SMI_PHYA: begin
                            mdio_out <= regs[7][5 + smi_bit_cnt];
                            if (smi_bit_cnt == 0) begin
                                smi_state <= SMI_REGA;
                                smi_bit_cnt <= 4;
                            end else smi_bit_cnt <= smi_bit_cnt - 1;
                        end
                        SMI_REGA: begin
                            mdio_out <= regs[7][0 + smi_bit_cnt];
                            if (smi_bit_cnt == 0) begin
                                smi_state <= SMI_TA;
                                smi_bit_cnt <= 1;
                            end else smi_bit_cnt <= smi_bit_cnt - 1;
                        end
                        SMI_TA: begin
                            if (smi_write_op) begin
                                mdio_out <= (smi_bit_cnt == 1) ? 1 : 0;
                            end else begin
                                mdio_oe <= 0;
                            end
                            if (smi_bit_cnt == 0) begin
                                smi_state <= SMI_DATA;
                                smi_bit_cnt <= 15;
                            end else smi_bit_cnt <= smi_bit_cnt - 1;
                        end
                        SMI_DATA: begin
                            if (smi_write_op) begin
                                mdio_out <= regs[8][smi_bit_cnt];
                            end else begin
                                // Read handled on rising edge
                            end
                            if (smi_bit_cnt == 0) begin
                                smi_state <= SMI_IDLE;
                            end else smi_bit_cnt <= smi_bit_cnt - 1;
                        end
                    endcase
                end else begin
                    if (smi_state == SMI_DATA && !smi_write_op) begin
                         regs[8][smi_bit_cnt] <= mdio_in;
                    end
                end
            end
        end
    end
    
    // AHB Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hready_out <= 1;
            hresp <= 0;
            regs[0] <= 0; 
            regs[1] <= 1; 
            regs[6] <= 0; 
            irq <= 0;
            tx_wr_ptr <= 0;
            tx_rd_ptr <= 0;
            rx_wr_ptr <= 0;
            rx_rd_ptr <= 0;
        end else begin
            hready_out <= 1;
            hresp <= 0;
            
            regs[2] <= ptp_counter[31:0];
            regs[3] <= ptp_counter[63:32];
            
            if (hsel && hready_in && htrans[1]) begin
                if (hwrite) begin
                    case (reg_idx)
                        4'h0: regs[0] <= hwdata;
                        4'h4: begin // TX_DATA FIFO
                             tx_fifo[tx_wr_ptr] <= hwdata;
                             tx_wr_ptr <= tx_wr_ptr + 1;
                             
                             if (regs[0][1]) begin // Loopback
                                 rx_fifo[rx_wr_ptr] <= hwdata;
                                 rx_wr_ptr <= rx_wr_ptr + 1;
                                 regs[10] <= regs[10] + 1; // RX Count
                                 // Simple latency calc: just current time
                                 regs[11] <= ptp_counter[31:0]; 
                             end
                        end
                        4'h6: regs[6] <= hwdata;
                        4'h7: regs[7] <= hwdata;
                        4'h8: regs[8] <= hwdata;
                        4'hC: regs[12] <= hwdata; // REFLEX_MATCH (Map to 0x30 in decoding, but here 4'hC = 12)
                        4'hD: regs[13] <= hwdata; // REFLEX_MASK
                        4'hE: regs[14] <= hwdata; // REFLEX_CTRL
                        default: ;
                    endcase
                end else begin
                    if (reg_idx == 4'h5) begin // RX_DATA FIFO
                        hrdata <= rx_fifo[rx_rd_ptr];
                        rx_rd_ptr <= rx_rd_ptr + 1;
                    end else begin
                        hrdata <= regs[reg_idx];
                    end
                end
            end
        end
    end

endmodule
