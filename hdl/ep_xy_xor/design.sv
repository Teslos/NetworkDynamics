// ===========================================================================
// design.sv  -  Equilibrium-Propagation training of an XY / Kuramoto XOR gate
//               Synthesizable-style fixed-point SystemVerilog (DUT).
//
// Port of notebooks/EP-XY-Network-Claude.jl.  Full EP TRAINING on-chip:
//   free relax (beta=0) -> +beta relax -> -beta relax -> EP gradient -> SGD+clip.
//
// N = 5 oscillators, all-to-all SYMMETRIC coupling (W=W', zero diagonal).
//   nodes 0,1 = inputs (clamped);  node 4 = output (nudged/read);  2,3 = hidden.
//   free-relaxing nodes = {2,3,4}.
//
// Fixed-point:
//   * phases/psi/target : 16-bit UNSIGNED binary-radians (brad), full circle
//     2*pi == 2**16.  +pi/2 = 0x4000 (TRUE), -pi/2 = 0xC000 (FALSE).  Angle
//     differences are 16-bit modular subtraction; sin/cos via brad-indexed ROM.
//   * W, h, gradients   : signed Q6.12  (1.0 == 4096).
//   * sin/cos ROM       : signed Q1.15  (1.0 == 32767).
//
// The sin & tanhalf ROMs are combinational FUNCTIONS backed by arrays filled
// once (via $sin/$tan in an initial block -- a portable sim technique; for real
// synthesis load them from a $readmemh mem-file).  All scaling constants are
// tunable localparams (fixed-point EP convergence is sensitive -- see README).
// ===========================================================================
`timescale 1ns/1ps
module ep_xy_xor_top #(
    parameter int N_EPOCH   = 400,     // training epochs   (reference uses 5000)
    parameter int MAX_ITERS = 200,     // Euler steps per relaxation cap
    parameter int STEP_SH   = 17,      // Euler: dphi_brad = (F * STEP_MUL) >>> STEP_SH
    parameter int STEP_MUL  = 1,       //   (folds dt=0.1 and value->brad = 2^16/2pi)
    parameter int NUDGE_SH  = 3,       // align beta*tanhalf (Q.24) to force Q.27
    parameter int GSC       = 51200,   // grad -> Q6.12 : g_q12 = (sum_cos * GSC)>>>15
    parameter int ETA_SH    = 4,       // W/h update: param -= g_clip >>> ETA_SH  (eta~1/16)
    parameter int PSI_SH    = 2,       // psi update shift (brad; larger step)
    parameter int STEADY    = 6        // convergence: max|dphi_brad| < STEADY
)(
    input  logic         clk,
    input  logic         rst,
    input  logic         start,
    output logic         busy,
    output logic         done,
    output logic [15:0]  epoch_o,
    output logic signed [31:0] cost_o,   // mean_p (1-cos)/2 * 4096
    output logic         cost_valid,
    output logic [3:0]   xor_out,
    output logic         xor_valid,
    // learned-parameter readout (valid once `done`)
    output logic signed [17:0] W_o   [0:4][0:4],   // Q6.12 symmetric coupling
    output logic signed [17:0] h_o   [0:4],         // Q6.12 field magnitude
    output logic [15:0]        psi_o [0:4]          // brad preferred phase
);
    // ---------------- ROMs (combinational lookup functions) ----------------
    localparam int LN = 1024;                 // ROM entries (10-bit index)
    logic signed [15:0] sinrom [0:LN-1];      // Q1.15
    logic signed [17:0] tanrom [0:LN-1];      // Q6.12, saturated
    initial begin
        real a, t;
        for (int i = 0; i < LN; i++) begin
            a = 2.0*3.14159265358979*i/LN;
            sinrom[i] = $rtoi($sin(a) * 32767.0);
            t = $tan(a/2.0) * 4096.0;
            if (t >  16.0*4096.0) t =  16.0*4096.0;   // saturate near d=pi
            if (t < -16.0*4096.0) t = -16.0*4096.0;
            tanrom[i] = $rtoi(t);
        end
    end
    function automatic logic signed [15:0] f_sin(input logic [15:0] b);
        return sinrom[b[15:6]];
    endfunction
    function automatic logic signed [15:0] f_cos(input logic [15:0] b);
        logic [15:0] c; c = b + 16'h4000; return sinrom[c[15:6]];   // cos = sin(x+pi/2)
    endfunction
    function automatic logic signed [17:0] f_tanh(input logic [15:0] b);
        return tanrom[b[15:6]];
    endfunction

    // ---------------- XOR data (brad) ----------------
    localparam logic [15:0] LO = 16'hC000, HI = 16'h4000;   // FALSE / TRUE
    logic [15:0] in1 [0:3], in2 [0:3], tgt [0:3];
    initial begin
        in1[0]=LO; in2[0]=LO; tgt[0]=LO;   // F,F -> F
        in1[1]=LO; in2[1]=HI; tgt[1]=HI;   // F,T -> T
        in1[2]=HI; in2[2]=LO; tgt[2]=HI;   // T,F -> T
        in1[3]=HI; in2[3]=HI; tgt[3]=LO;   // T,T -> F
    end

    // ---------------- trained state ----------------
    logic [15:0]        phase [0:4];
    logic [15:0]        phi0 [0:4][0:3], phiP [0:4][0:3], phiN [0:4][0:3];
    logic signed [17:0] W [0:4][0:4];        // Q6.12 symmetric, zero diag
    logic signed [17:0] hf [0:4];            // Q6.12
    logic [15:0]        psi [0:4];           // brad
    logic signed [31:0] gW [0:4][0:4], ghf [0:4], gpsi [0:4];

    // ---------------- LFSR noise (per-epoch free-node re-init) ----------------
    logic [31:0] lfsr;
    always_ff @(posedge clk)
        if (rst) lfsr <= 32'h1234_5678;
        else     lfsr <= {lfsr[30:0],1'b0} ^ (lfsr[31] ? 32'h04C11DB7 : 32'h0);
    function automatic logic [15:0] noise16(input logic [31:0] r);
        // ~ +/- 2048 brad  (~ +/- 0.2 rad)
        return {{4{r[11]}}, r[11:0]};
    endfunction

    localparam int OUT = 4;
    localparam logic signed [17:0] BETA = 18'sd41;   // 0.01 in Q6.12

    // learned-parameter readout (mirror internal state to output ports)
    always_comb begin
        for (int a=0;a<5;a++) begin
            h_o[a]=hf[a]; psi_o[a]=psi[a];
            for (int b=0;b<5;b++) W_o[a][b]=W[a][b];
        end
    end

    // ---------------- FSM ----------------
    typedef enum logic [4:0] {
        S_IDLE,S_INIT,S_LOAD,S_STEP,S_ITER,S_STORE,S_NEXTP,
        S_GRAD,S_UPD,S_EEND,S_ELOAD,S_ESTEP,S_ESTORE,S_DONE
    } state_t;
    state_t st;
    logic [15:0] epoch, iter;
    logic [1:0]  pat, kind;                   // kind: 0 free, 1 +beta, 2 -beta
    logic signed [17:0] cur_beta;
    logic [15:0] tcur;
    logic [15:0] dmax;
    integer jn;

    // combinational single-node force (Q.27), given node index j
    function automatic logic signed [39:0] node_force(input int j, input logic signed [17:0] beta_out);
        logic signed [39:0] acc; int k;
        acc = 40'sd0;
        for (k=0;k<5;k++) acc -= ($signed({{22{W[k][j][17]}},W[k][j]}) * f_sin(phase[j]-phase[k]));
        acc -= ($signed({{22{hf[j][17]}},hf[j]}) * f_sin(phase[j]-psi[j]));      // field
        if (j==OUT && beta_out!=0)
            acc -= (($signed({{22{beta_out[17]}},beta_out}) * f_tanh(phase[j]-tcur)) <<< NUDGE_SH);
        return acc;
    endfunction
    function automatic logic signed [15:0] step_brad(input logic signed [39:0] f);
        logic signed [39:0] t; t = (f * STEP_MUL) >>> STEP_SH;
        if (t> 30000) t= 30000; if (t< -30000) t=-30000; return t[15:0];
    endfunction
    function automatic logic signed [31:0] clip_q12(input logic signed [31:0] g);
        if (g >  (4<<12)) return  (4<<12);
        if (g < -(4<<12)) return -(4<<12);
        return g;
    endfunction

    always_ff @(posedge clk) begin
        if (rst) begin
            st<=S_IDLE; busy<=0; done<=0; cost_valid<=0; xor_valid<=0;
            epoch<=0; xor_out<=0; cost_o<=0; epoch_o<=0;
        end else begin
            cost_valid<=0;
            case (st)
            S_IDLE: begin done<=0; xor_valid<=0; if (start) begin busy<=1; st<=S_INIT; end end

            S_INIT: begin
                for (int a=0;a<5;a++) begin
                    hf[a]  <= 18'sd8;                          // ~0.002
                    psi[a] <= 16'(a) * 16'h1000;              // spread
                    for (int b=0;b<5;b++) W[a][b] <= (a==b)?18'sd0:18'sd12;  // ~0.003 sym
                end
                epoch<=0; pat<=0; kind<=0; st<=S_LOAD;
            end

            // ---- load a relaxation ----
            S_LOAD: begin
                phase[0]<=in1[pat]; phase[1]<=in2[pat]; tcur<=tgt[pat];
                if (kind==0) begin
                    phase[2]<=noise16(lfsr);
                    phase[3]<=noise16(lfsr ^ 32'hA5A5A5A5);
                    phase[4]<=noise16(lfsr ^ 32'h5A5A5A5A);
                    cur_beta<=18'sd0;
                end else begin
                    phase[2]<=phi0[2][pat]; phase[3]<=phi0[3][pat]; phase[4]<=phi0[4][pat];
                    cur_beta<=(kind==1)? BETA : -BETA;
                end
                iter<=0; jn<=2; dmax<=0; st<=S_STEP;
            end
            // ---- one node per cycle (Gauss-Seidel Euler sweep) ----
            S_STEP: begin
                logic signed [15:0] db;
                db = step_brad(node_force(jn, cur_beta));
                phase[jn] <= phase[jn] + db;
                if ((db[15]?-db:db) > dmax) dmax <= (db[15]?-db:db);
                if (jn==4) st<=S_ITER; else jn<=jn+1;
            end
            S_ITER: begin
                iter<=iter+1;
                if (dmax<STEADY || iter+1>=MAX_ITERS) st<=S_STORE;
                else begin jn<=2; dmax<=0; st<=S_STEP; end
            end
            S_STORE: begin
                for (int a=0;a<5;a++) begin
                    if (kind==0) phi0[a][pat]<=phase[a];
                    else if (kind==1) phiP[a][pat]<=phase[a];
                    else phiN[a][pat]<=phase[a];
                end
                if (kind==2) st<=S_NEXTP; else begin kind<=kind+1; st<=S_LOAD; end
            end
            S_NEXTP: begin
                kind<=0;
                if (pat==3) begin pat<=0; st<=S_GRAD; end
                else begin pat<=pat+1; st<=S_LOAD; end
            end

            // ---- EP gradients over all 4 patterns (combinational, one cycle) ----
            S_GRAD: begin
                logic signed [31:0] cw, sw, ca;
                for (int a=0;a<5;a++) begin ghf[a]=0; gpsi[a]=0;
                    for (int b=0;b<5;b++) gW[a][b]=0; end
                ca=0;
                for (int p=0;p<4;p++) begin
                    for (int a=0;a<5;a++) begin
                        for (int b=0;b<5;b++) begin
                            cw = $signed(f_cos(phiN[a][p]-phiN[b][p]))
                               - $signed(f_cos(phiP[a][p]-phiP[b][p]));
                            gW[a][b] = gW[a][b] + cw;
                        end
                        cw = $signed(f_cos(phiN[a][p]-psi[a])) - $signed(f_cos(phiP[a][p]-psi[a]));
                        ghf[a] = ghf[a] + cw;
                        sw = $signed(f_sin(phiN[a][p]-psi[a])) - $signed(f_sin(phiP[a][p]-psi[a]));
                        gpsi[a] = gpsi[a] + (($signed({{14{hf[a][17]}},hf[a]}) * sw) >>> 12);
                    end
                    ca = ca + (32'sd32767 - $signed(f_cos(phi0[OUT][p]-tgt[p])));  // (1-cos) Q1.15
                end
                cost_o <= (ca <<< 12) / (4*32767);       // mean (1-cos)/2 * 4096
                st<=S_UPD;
            end
            // ---- SGD + clip update ----
            S_UPD: begin
                for (int a=0;a<5;a++) begin
                    logic signed [31:0] gh, gp;
                    gh = clip_q12(($signed(ghf[a]) * GSC) >>> 15);
                    hf[a] <= hf[a] - 18'(gh >>> ETA_SH);
                    gp = clip_q12(($signed(gpsi[a]) * GSC) >>> 15);
                    psi[a] <= psi[a] - 16'(gp >>> PSI_SH);   // param -= eta*grad
                    for (int b=0;b<5;b++) if (a<b) begin
                        logic signed [31:0] gw;
                        gw = clip_q12(($signed(gW[a][b]) * GSC) >>> 15);
                        W[a][b] <= W[a][b] - 18'(gw >>> ETA_SH);
                        W[b][a] <= W[a][b] - 18'(gw >>> ETA_SH);
                    end
                end
                st<=S_EEND;
            end
            S_EEND: begin
                epoch_o<=epoch; cost_valid<=1;
                if (epoch+1>=N_EPOCH) begin pat<=0; kind<=0; st<=S_ELOAD; end
                else begin epoch<=epoch+1; pat<=0; kind<=0; st<=S_LOAD; end
            end

            // ---- evaluation: free relax, read output sign ----
            S_ELOAD: begin
                phase[0]<=in1[pat]; phase[1]<=in2[pat];
                phase[2]<=0; phase[3]<=0; phase[4]<=0;
                cur_beta<=0; tcur<=tgt[pat]; iter<=0; jn<=2; dmax<=0; st<=S_ESTEP;
            end
            S_ESTEP: begin
                logic signed [15:0] db;
                db = step_brad(node_force(jn, 18'sd0));
                phase[jn]<=phase[jn]+db;
                if ((db[15]?-db:db)>dmax) dmax<=(db[15]?-db:db);
                if (jn==4) begin
                    iter<=iter+1;
                    if (dmax<STEADY || iter+1>=MAX_ITERS) st<=S_ESTORE;
                    else begin jn<=2; dmax<=0; end
                end else jn<=jn+1;
            end
            S_ESTORE: begin
                xor_out[pat] <= ~phase[OUT][15];        // TRUE if upper half (~+pi/2)
                if (pat==3) begin xor_valid<=1; done<=1; busy<=0; st<=S_DONE; end
                else begin pat<=pat+1; st<=S_ELOAD; end
            end
            S_DONE: if (!start) st<=S_IDLE;
            default: st<=S_IDLE;
            endcase
        end
    end
endmodule
