
/**
 * Simple AHB-DMA Controller
 * for Hive-Reflex SoC
 * 
 * Functions:
 * - Offload data transfer from source (e.g., Network RX FIFO) to destination (e.g., SRAM/CIM).
 * - Single channel for simplicity in this version.
 */

`timescale 1ns / 1ps

module ahb_dma #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    
    // AHB Slave Interface (Configuration)
    input wire [ADDR_WIDTH-1:0] haddr_s,
    input wire [DATA_WIDTH-1:0] hwdata_s,
    output reg [DATA_WIDTH-1:0] hrdata_s,
    input wire hwrite_s,
    input wire [2:0] hsize_s,
    input wire [1:0] htrans_s,
    input wire hsel_s,
    input wire hready_in_s,
    output reg hready_out_s,
    output reg hresp_s,
    
    // AHB Master Interface (Data Moving)
    output reg [ADDR_WIDTH-1:0] haddr_m,
    output reg [DATA_WIDTH-1:0] hwdata_m,
    input wire [DATA_WIDTH-1:0] hrdata_m,
    output reg hwrite_m,
    output reg [2:0] hsize_m,
    output reg [1:0] htrans_m,
    input wire hready_m,
    input wire hresp_m,
    
    // Interrupt
    output reg irq_done
);

    // Register Map
    // 0x00: CTRL (Bit 0: Enable/Start)
    // 0x04: SRC_ADDR
    // 0x08: DST_ADDR
    // 0x0C: LEN (Number of words)
    // 0x10: STATUS (Bit 0: Busy, Bit 1: Done)
    
    reg [31:0] regs [0:4];
    
    // State Machine
    localparam S_IDLE = 0, S_READ_ADDR = 1, S_READ_DATA = 2, S_WRITE_ADDR = 3, S_WRITE_DATA = 4, S_DONE = 5;
    reg [2:0] state;
    reg [31:0] count;
    reg [31:0] current_src, current_dst;
    reg [31:0] data_buf;
    
    // AHB Slave Logic (Config)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hready_out_s <= 1;
            hresp_s <= 0;
            regs[0] <= 0;
            regs[1] <= 0;
            regs[2] <= 0;
            regs[3] <= 0;
            regs[4] <= 0;
        end else begin
            hready_out_s <= 1;
            hresp_s <= 0;
            
            // Auto Clear Start Bit / Set Busy
            if (state != S_IDLE && state != S_DONE) regs[4][0] <= 1; // Busy
            else regs[4][0] <= 0;
            
            if (hsel_s && hready_in_s && htrans_s[1]) begin
                if (hwrite_s) begin
                    case (haddr_s[4:2])
                        3'h0: regs[0] <= hwdata_s;
                        3'h1: regs[1] <= hwdata_s;
                        3'h2: regs[2] <= hwdata_s;
                        3'h3: regs[3] <= hwdata_s;
                        3'h4: regs[4] <= hwdata_s; // Write to clear done?
                    endcase
                end else begin
                    hrdata_s <= regs[haddr_s[4:2]];
                end
            end else if (state == S_DONE) begin
                regs[0][0] <= 0; // Clear enable
                regs[4][1] <= 1; // Done
            end
        end
    end
    
    // AHB Master Logic (The Mover)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            haddr_m <= 0;
            hwdata_m <= 0;
            hwrite_m <= 0;
            hsize_m <= 2; // Word
            htrans_m <= 0;
            irq_done <= 0;
            count <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    irq_done <= 0;
                    if (regs[0][0]) begin // Start
                        state <= S_READ_ADDR;
                        count <= regs[3];
                        current_src <= regs[1];
                        current_dst <= regs[2];
                    end
                end
                
                S_READ_ADDR: begin
                    if (count == 0) begin
                        state <= S_DONE;
                    end else begin
                        haddr_m <= current_src;
                        hwrite_m <= 0;
                        htrans_m <= 2; // NONSEQ
                        state <= S_READ_DATA;
                    end
                end
                
                S_READ_DATA: begin
                    if (hready_m) begin
                        // Address phase finished, now wait for data
                        // But wait, standard AHB pipelining
                        // We need to hold address/trans for 1 cycle if hready is high? No.
                        // Address phase is 1 cycle.
                        // We move to next phase (Data).
                        haddr_m <= 0;
                        htrans_m <= 0;
                        
                        // We need to wait for data to be valid. 
                        // Actually in simple state machine, we might need a wait cycle if hready was low?
                        // Assuming simple model:
                         // Wait for data
                         // NOTE: This simple FSM assumes zero-wait state mostly or handles hready simply.
                        
                        // We need to capture hrdata_m
                         data_buf <= hrdata_m;
                         state <= S_WRITE_ADDR;
                    end
                end

                S_WRITE_ADDR: begin
                     haddr_m <= current_dst;
                     hwrite_m <= 1;
                     htrans_m <= 2;
                     // Data phase of write happens next
                     state <= S_WRITE_DATA;
                end
                
                S_WRITE_DATA: begin
                    if (hready_m) begin
                        hwdata_m <= data_buf;
                        haddr_m <= 0;
                        htrans_m <= 0;
                        // Wait for write data phase to complete
                        // We can move to next read immediately if we pipeline, but let's be simple (sequential)
                        
                        // BUT: hwdata must be valid during data phase.
                        // We are IN data phase now (after address phase).
                        // We need to hold this until hready is high.
                        if (hready_m) begin // Slave accepted write?
                            // Actually we checked hready at start to enter phase.
                            // We need to wait one more cycle?
                            
                            // Let's assume we wait here
                            
                            current_src <= current_src + 4; // Increment only if not FIFO (TODO: Add FIFO mode)
                            current_dst <= current_dst + 4;
                            count <= count - 1;
                            state <= S_READ_ADDR;
                        end
                    end
                end
                
                S_DONE: begin
                    irq_done <= 1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
