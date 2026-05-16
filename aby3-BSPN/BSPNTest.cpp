#include "BSPNTest.h"
#include "BSPN.h"
#include <aby3-Basic/BuildingBlocks.h>
#include <fstream>
#include <iostream>

using namespace aby3;
using namespace oc;

void BSPN_test(const oc::CLP& cmd) {
    u64 partyIdx = cmd.getOr<int>("role", -1);
    if(partyIdx == -1) {
        throw std::runtime_error("role not set");
    }
    
    IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    basic_setup(partyIdx, ios, enc, eval, runtime);

    // Create a dummy BSPN file (only P0 needs to do this really, or all if reading locally)
    std::string filename = "test_bspn.txt";
    if (partyIdx == 0) {
        std::ofstream out(filename);
        
        // Define Data: 20 rows total.
        // Cluster A (Rows 0-11, 60%): Age [20-30], Salary [30k-50k]
        // Cluster B (Rows 12-19, 40%): Age [50-60], Salary [80k-100k]
        
        // Header
        out << "model_type=BSPN_Full_Export\n";
        out << "num_trees=1\n\n";
        out << "Tree=0\n";
        out << "num_nodes=7\n"; // Root, Prod1, Prod2, 4 Leaves

        // Root: Sum (id=0)
        // Children: Product 1 (id=1), Product 2 (id=2)
        // Weights: 0.6, 0.4
        out << "id=0 type=Sum scope=0,1 cardinality=20 children=1,2 weights=0.6,0.4\n";
        
        // --- Branch A ---
        out << "id=1 type=Product scope=0,1 cardinality=12 children=3,4\n";
        
        // Leaf 3: IdentityNumericLeaf for Age (Scope 0) in Cluster A
        // Interval: 20.0:30.0
        // IDs: 0,1,2,3,4,5,6,7,8,9,10,11
        std::string ids_a = "0|1|2|3|4|5|6|7|8|9|10|11";
                // --- Branch B ---
        out << "id=2 type=Product scope=0,1 cardinality=8 children=5,6\n";
        
        out << "id=3 type=IdentityNumericLeaf scope=0 cardinality=12 intervals=20.0:30.0 bitmaps=" << ids_a << "\n";

        // Leaf 4: IdentityNumericLeaf for Salary (Scope 1) in Cluster A
        // Interval: 30000.0:50000.0
        // IDs: 0,1,2,3,4,5,6,7,8,9,10,11
        out << "id=4 type=IdentityNumericLeaf scope=1 cardinality=12 intervals=30000.0:50000.0 bitmaps=" << ids_a << "\n";


        // Leaf 5: IdentityNumericLeaf for Age (Scope 0) in Cluster B
        // Interval: 50.0:60.0
        // IDs: 12,13,14,15,16,17,18,19
        std::string ids_b = "12|13|14|15|16|17|18|19";
        out << "id=5 type=IdentityNumericLeaf scope=0 cardinality=8 intervals=50.0:60.0 bitmaps=" << ids_b << "\n";

        // Leaf 6: IdentityNumericLeaf for Salary (Scope 1) in Cluster B
        // Interval: 80000.0:100000.0
        // IDs: 12,13,14,15,16,17,18,19
        out << "id=6 type=IdentityNumericLeaf scope=1 cardinality=8 intervals=80000.0:100000.0 bitmaps=" << ids_b << "\n";
        
        out.close();
        std::cout << "Created " << filename << std::endl;
    }
    
    // Sync to ensure file write finishes
    runtime.mComm.mNext.asyncSendCopy(partyIdx);
    u64 tmp;
    runtime.mComm.mPrev.recv(tmp);

    PlainBSPN p_bspn;
    p_bspn.load_from_file(filename);

    SecureBSPN s_bspn;
    
    std::cout << "Encrypting BSPN..." << std::endl;
    s_bspn.encrypt(p_bspn, 0, enc, runtime); // 0 is the source party
    std::cout << "Encryption done." << std::endl;

    // -------------------------------------------------------------------
    // Query 1: COUNT(*) WHERE Age >= 40
    // -------------------------------------------------------------------
    // We expect this to match only Cluster B (Age [50-60]).
    // Cluster A is [20-30].
    // So probability should be 0.4.
    // Count = 0.4 * 20 = 8.
    
    std::cout << "\n--- Query 1: Probability(Age >= 40) ---" << std::endl;
    
    // Mocking Evidence: Age (Scope 0) in [40, 100]
    std::map<int, std::vector<double>> evidence_ranges;
    evidence_ranges[0] = {40.0, 100.0}; 
    std::map<int, std::vector<int>> evidence_inclusive; // Not used yet in mockup

    int feature_scope_dummy = -1; // Not computing expectation of a specific feature, just probability of evidence
    std::vector<int> relevant_scope = {0};

    // Note: In a real SPN, P(E) is computed by Sum/Product logic bottom-up.
    // Our compute_expectation function is currently a placeholder for full inference.
    // But let's assume it runs the circuit.
    
    sf64<D16> res_share = s_bspn.compute_expectation(feature_scope_dummy, relevant_scope, evidence_ranges, evidence_inclusive, enc, eval, runtime);
    
    f64<D16> res_plain;
    enc.revealAll(runtime, res_share, res_plain).get();
    
    std::cout << "Result (Scaled 65536): " << res_plain.mValue << std::endl;
    std::cout << "Value: " << static_cast<double>(res_plain) << std::endl;
    std::cout << "Expected: ~0.4" << std::endl;
    
    // -------------------------------------------------------------------
    // Query 2: AVG(Salary) (Global Expectation)
    // -------------------------------------------------------------------
    // We want E[Salary].
    // No evidence (or evidence covers all rows).
    // Cluster A (0.6) * Mid(30k, 50k) + Cluster B (0.4) * Mid(80k, 100k)
    // = 0.6 * 40k + 0.4 * 90k = 24k + 36k = 60k.
    
    std::cout << "\n--- Query 2: Expectation(Salary) ---" << std::endl;
    
    // Clear evidence for global expectation
    evidence_ranges.clear();
    int feature_scope_salary = 1;
    relevant_scope = {1};

    // We need to support "Leaf Expectation" in the C++ code for this to work.
    // Currently BSPN.cpp Leaf logic returns 1.0 (placeholder).
    // To make this test pass meaningfully, we'd need to implement the leaf logic 
    // to return the midpoint of the interval.
    // But for now, we just run it to ensure no crash and check the structure flow.
    
    res_share = s_bspn.compute_expectation(feature_scope_salary, relevant_scope, evidence_ranges, evidence_inclusive, enc, eval, runtime);
    enc.revealAll(runtime, res_share, res_plain).get();
    
    std::cout << "Result (Scaled 65536): " << res_plain.mValue << std::endl;
    std::cout << "Value: " << static_cast<double>(res_plain) << std::endl;
    std::cout << "Expected: ~60000.0" << std::endl;

    std::cout << "\nTest Execution Finished." << std::endl;
}
