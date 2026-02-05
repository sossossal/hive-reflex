
// Mocks for missing/placeholder modules in Hive-Reflex Top

module riscv_cpu_wrapper (
    input clk,
    input rst_n,
    output reg [31:0] haddr,
    output reg [31:0] hwdata,
    input [31:0] hrdata,
    output reg hwrite,
    output reg [2:0] hsize,
    output reg [1:0] htrans,
    input hready,
    input hresp,
    input jtag_tck,
    input jtag_tms,
    input jtag_tdi,
    output jtag_tdo
);

    // Initial State
    initial begin
        haddr = 0;
        hwdata = 0;
        hwrite = 0;
        hsize = 2; // 32-bit
        htrans = 0; // IDLE
    end

    // AHB Write Task
    task ahb_write(input [31:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            while (!hready) @(posedge clk);
            
            // Address Phase
            haddr <= addr;
            hwrite <= 1;
            hsize <= 2; // Word
            htrans <= 2; // NONSEQ
            
            @(posedge clk);
            while (!hready) @(posedge clk);
            
            // Data Phase
            hwdata <= data;
            haddr <= 0;
            htrans <= 0; // IDLE
            hwrite <= 0;
            
            @(posedge clk);
            while (!hready) @(posedge clk);
        end
    endtask

    // AHB Read Task
    task ahb_read(input [31:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            while (!hready) @(posedge clk);
            
            // Address Phase
            haddr <= addr;
            hwrite <= 0;
            hsize <= 2;
            htrans <= 2; // NONSEQ
            
            @(posedge clk);
            while (!hready) @(posedge clk);
            
            // Data Phase
            haddr <= 0;
            htrans <= 0;
            
            // Sample Data
            // Wait for slave to be ready with data
             @(posedge clk); // wait one cycle for data phase to complete? 
                             // AHB pipelining: Data is valid when HREADY is high in data phase
             // Actually, in the cycle after Address Phase, we are in Data Phase.
             // If HREADY was high at end of Addr Phase, we enter Data Phase.
             // We need to wait until HREADY is high AGAIN to sample data.
             // The previous wait loop handled the Addr Phase wait.
             
             // Now we are in Data Phase.
             while (!hready) @(posedge clk);
             data = hrdata;
        end
    endtask

endmodule

module peripheral_subsystem (
    input clk,
    input rst_n,
    input [31:0] haddr,
    input [31:0] hwdata,
    output [31:0] hrdata,
    input hwrite,
    input hsel,
    output hready,
    output hresp,
    output [3:0] led,
    output uart_tx,
    input uart_rx
);
endmodule

// Ensure cim_core_top is also available if not in file list, 
// but it was in the file list.

module IBUFDS (
    input I,
    input IB,
    output O
);
    assign O = I;
endmodule

module ahb_sram #(
    parameter ADDR_WIDTH = 19,
    parameter DATA_WIDTH = 32
) (
    input clk,
    input rst_n,
    input [31:0] haddr,
    input [31:0] hwdata,
    output [31:0] hrdata,
    input hwrite,
    input hsel,
    output hready,
    output hresp
);
endmodule

module IOBUF (
    input I,
    input T,
    output O,
    inout IO
);
    assign O = IO;
    assign IO = (T == 0) ? I : 1'bz;
endmodule
