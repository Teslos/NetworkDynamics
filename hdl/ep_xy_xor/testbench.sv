// ===========================================================================
// testbench.sv  -  self-checking TB for the EP-XY-XOR trainer.
//
// Contains a behavioral `real` GOLDEN REFERENCE (a direct transcription of
// notebooks/EP-XY-Network-Claude.jl, using $sin/$cos) that trains EP-XY-XOR in
// software-style math, prints its cost history + XOR truth table, and is used
// to sanity-check the fixed-point DUT.  The golden model is simulation-only.
//
// Flow:
//   1. run the golden reference (fast, real math) -> prints cost + XOR.
//   2. run the fixed-point DUT      -> prints cost + XOR.
//   3. self-check DUT XOR vs the expected XOR truth table, report PASS/FAIL.
//
// EDA Playground: Language=SystemVerilog, Simulator=Aldec Riviera-PRO (real/$sin
// support; Icarus Verilog also works), tick "Open EPWave after run".
// ===========================================================================
`timescale 1ns/1ps
module tb;

    // ---------------- clock / reset ----------------
    logic clk = 0, rst = 1, start = 0;
    always #5 clk = ~clk;

    // ---------------- DUT ----------------
    logic        busy, done, cost_valid, xor_valid;
    logic [15:0] epoch_o;
    logic signed [31:0] cost_o;
    logic [3:0]  xor_out;
    logic signed [17:0] W_o [0:4][0:4];
    logic signed [17:0] h_o [0:4];
    logic [15:0]        psi_o [0:4];

    ep_xy_xor_top #(.N_EPOCH(300), .MAX_ITERS(150)) dut (
        .clk(clk), .rst(rst), .start(start),
        .busy(busy), .done(done),
        .epoch_o(epoch_o), .cost_o(cost_o), .cost_valid(cost_valid),
        .xor_out(xor_out), .xor_valid(xor_valid),
        .W_o(W_o), .h_o(h_o), .psi_o(psi_o)
    );

    // brad (16-bit binary radians, wrapped) -> radians in (-pi, pi]
    function automatic real brad2rad(input logic [15:0] b);
        real v; v = (b >= 16'h8000) ? (real'(b) - 65536.0) : real'(b);
        return v * 2.0*3.14159265358979 / 65536.0;
    endfunction

    // expected XOR per pattern (bit p): F,T,T,F -> 0,1,1,0
    localparam logic [3:0] EXPECTED = 4'b0110;

    // print DUT cost as it trains
    always @(posedge clk)
        if (cost_valid && (epoch_o % 50 == 0))
            $display("[DUT]  epoch %0d   cost = %0.5f", epoch_o, $itor(cost_o)/4096.0);

    // =====================================================================
    // GOLDEN REFERENCE  (behavioral, real)  -- faithful to the Julia code
    // =====================================================================
    localparam real PI   = 3.14159265358979;
    localparam int  RN    = 5;                 // nodes
    localparam int  ROUT  = 4;                 // output node (0-indexed)
    localparam int  RIN0=0, RIN1=1;            // input nodes
    localparam real RBETA = 0.01;              // nudge strength
    localparam real RETA  = 0.05;              // SGD learning rate
    localparam real RDT   = 0.1;               // Euler step
    localparam int  RITERS= 400;               // max Euler steps / relaxation
    localparam real RTOL  = 1.0e-4;            // steady-state tol on max|F|
    localparam int  REPOCH= 1000;              // epochs (raise if it converges but XOR imperfect)

    real Wr [0:RN-1][0:RN-1];
    real hr [0:RN-1];
    real pr [0:RN-1];                          // psi
    real ph [0:RN-1];                          // working phases
    real p0 [0:RN-1][0:3];                     // free equilibria / pattern
    real pp [0:RN-1][0:3];                     // +beta equilibria
    real pn [0:RN-1][0:3];                     // -beta equilibria
    real rin1 [0:3];  real rin2 [0:3];  real rtgt [0:3];
    real gWr[0:RN-1][0:RN-1];  real ghr[0:RN-1];  real gpr[0:RN-1];

    function automatic real clampf(input real x, input real lo, input real hi);
        if (x<lo) return lo; else if (x>hi) return hi; else return x;
    endfunction

    // one Euler relaxation of `ph` toward equilibrium (inputs held fixed)
    task automatic xy_relax(input real tgt, input real beta);
        real F [0:RN-1];
        real d, mx, acc;
        int  it, j, k;
        for (it=0; it<RITERS; it++) begin
            for (j=0;j<RN;j++) begin
                acc = 0.0;
                for (k=0;k<RN;k++) acc += Wr[k][j]*$sin(ph[j]-ph[k]);
                F[j] = -acc - hr[j]*$sin(ph[j]-pr[j]);
            end
            if (beta != 0.0) begin
                d = ph[ROUT]-tgt;
                F[ROUT] -= beta*$sin(d)/(1.0+$cos(d)+1.0e-10);
            end
            F[RIN0]=0.0; F[RIN1]=0.0;               // clamp inputs
            mx = 0.0;
            for (j=0;j<RN;j++) begin
                ph[j] += RDT*F[j];
                if ((F[j]<0?-F[j]:F[j]) > mx) mx = (F[j]<0?-F[j]:F[j]);
            end
            if (mx < RTOL) break;
        end
    endtask

    real gauss_seed;
    function automatic real smallnoise();  // ~0.1*randn stand-in
        gauss_seed = gauss_seed*1103515245.0 + 12345.0;
        gauss_seed = gauss_seed - $floor(gauss_seed/2147483648.0)*2147483648.0;
        return 0.1*((gauss_seed/2147483648.0)-0.5);
    endfunction

    task automatic run_golden();
        int e,p,j,k;
        real cost, scale;
        // data
        rin1[0]=-PI/2; rin2[0]=-PI/2; rtgt[0]=-PI/2;
        rin1[1]=-PI/2; rin2[1]= PI/2; rtgt[1]= PI/2;
        rin1[2]= PI/2; rin2[2]=-PI/2; rtgt[2]= PI/2;
        rin1[3]= PI/2; rin2[3]= PI/2; rtgt[3]=-PI/2;
        // init params
        gauss_seed = 1234.0;
        for (j=0;j<RN;j++) begin
            hr[j]=0.002; pr[j]=2.0*PI*(j/5.0)-PI;
            for (k=0;k<RN;k++) Wr[j][k]=(j==k)?0.0:0.01;
        end
        scale = 2.0*RBETA;
        $display("---- GOLDEN REFERENCE (real) : EP-XY-XOR ----");
        for (e=1;e<=REPOCH;e++) begin
            for (j=0;j<RN;j++) begin ghr[j]=0; gpr[j]=0;
                for (k=0;k<RN;k++) gWr[j][k]=0; end
            cost = 0.0;
            for (p=0;p<4;p++) begin
                // free relax from clamped inputs + small noise on free nodes
                ph[RIN0]=rin1[p]; ph[RIN1]=rin2[p];
                ph[2]=smallnoise(); ph[3]=smallnoise(); ph[4]=smallnoise();
                xy_relax(rtgt[p], 0.0);
                for (j=0;j<RN;j++) p0[j][p]=ph[j];
                // +beta from free eq
                for (j=0;j<RN;j++) ph[j]=p0[j][p];
                xy_relax(rtgt[p],  RBETA);  for (j=0;j<RN;j++) pp[j][p]=ph[j];
                // -beta from free eq
                for (j=0;j<RN;j++) ph[j]=p0[j][p];
                xy_relax(rtgt[p], -RBETA);  for (j=0;j<RN;j++) pn[j][p]=ph[j];
                // cost (reported)
                cost += (1.0-$cos(p0[ROUT][p]-rtgt[p]))/2.0;
                // gradient accumulation
                for (j=0;j<RN;j++) begin
                    for (k=0;k<RN;k++)
                        gWr[j][k] += ($cos(pn[j][p]-pn[k][p]) - $cos(pp[j][p]-pp[k][p]))/4.0;
                    ghr[j] += ($cos(pn[j][p]-pr[j]) - $cos(pp[j][p]-pr[j]))/4.0;
                    gpr[j] += hr[j]*($sin(pn[j][p]-pr[j]) - $sin(pp[j][p]-pr[j]))/4.0;
                end
            end
            cost = cost/4.0;
            // SGD + clip (grad already /N_data; divide by scale, clip, step)
            for (j=0;j<RN;j++) begin
                hr[j] -= RETA*clampf(ghr[j]/scale,-1.0,1.0);
                pr[j] -= RETA*clampf(gpr[j]/scale,-1.0,1.0);
                for (k=0;k<RN;k++)
                    if (j<k) begin
                        real g; g = RETA*clampf(gWr[j][k]/scale,-1.0,1.0);
                        Wr[j][k]-=g; Wr[k][j]-=g;      // symmetric
                    end
            end
            if (e%250==0 || e==1)
                $display("[REF]  epoch %0d   cost = %0.5f", e, cost);
        end
        // evaluate XOR (free dynamics)
        $display("---- GOLDEN REFERENCE : XOR truth table ----");
        for (p=0;p<4;p++) begin
            ph[RIN0]=rin1[p]; ph[RIN1]=rin2[p]; ph[2]=0; ph[3]=0; ph[4]=0;
            xy_relax(rtgt[p], 0.0);
            $display("  pat %0d  in=(%s,%s)  phi_out=%f rad  -> %s (expect %s)",
                p, (rin1[p]>0)?"T":"F", (rin2[p]>0)?"T":"F", ph[ROUT],
                ($sin(ph[ROUT])>0)?"T":"F", EXPECTED[p]?"T":"F");
        end
    endtask

    // =====================================================================
    // Main
    // =====================================================================
    integer errors;
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, tb);

        // (1) golden reference
        run_golden();

        // (2) fixed-point DUT
        $display("\n---- FIXED-POINT DUT : training ----");
        rst = 1; start = 0;
        repeat (4) @(posedge clk);
        rst = 0; @(posedge clk);
        start = 1; @(posedge clk); start = 0;

        wait (done);
        @(posedge clk);

        // (3) self-check
        $display("---- FIXED-POINT DUT : XOR truth table ----");
        errors = 0;
        for (int p=0;p<4;p++) begin
            logic got, exp;
            got = xor_out[p]; exp = EXPECTED[p];
            $display("  pat %0d  ->  %s  (expect %s)  %s",
                     p, got?"T":"F", exp?"T":"F", (got===exp)?"PASS":"FAIL");
            if (got !== exp) errors++;
        end
        if (errors==0) $display("\n*** DUT XOR: ALL 4 PATTERNS PASS ***");
        else           $display("\n*** DUT XOR: %0d/4 FAIL (tune ETA_SH/STEP_MUL/GSC/N_EPOCH) ***", errors);

        // ---- learned parameters (fixed-point -> real) ----
        $display("\n---- DUT learned parameters ----");
        for (int a=0;a<5;a++)
            $display("  node %0d:  h=%f   psi=%f rad", a, $itor(h_o[a])/4096.0, brad2rad(psi_o[a]));
        $display("  symmetric coupling W (upper triangle):");
        for (int a=0;a<5;a++)
            for (int b=a+1;b<5;b++)
                $display("    W[%0d][%0d]=%f", a, b, $itor(W_o[a][b])/4096.0);

        $finish;
    end

    // safety timeout
    initial begin
        #200_000_000;
        $display("TIMEOUT");
        $finish;
    end
endmodule
