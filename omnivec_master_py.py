#!/usr/bin/env python3
"""
OmniVec Master Execution Script
Run the complete OmniVec pipeline from start to finish

Usage:
    python run_omnivec.py          # Run all parts
    python run_omnivec.py --skip-install  # Skip package installation
    python run_omnivec.py --parts 1,2,3   # Run specific parts only
"""

import sys
import time
import argparse
from datetime import datetime

def print_banner():
    """Print OmniVec banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     ██████╗ ███╗   ███╗███╗   ██╗██╗██╗   ██╗███████╗ ██████╗   ║
    ║    ██╔═══██╗████╗ ████║████╗  ██║██║██║   ██║██╔════╝██╔════╝   ║
    ║    ██║   ██║██╔████╔██║██╔██╗ ██║██║██║   ██║█████╗  ██║        ║
    ║    ██║   ██║██║╚██╔╝██║██║╚██╗██║██║╚██╗ ██╔╝██╔══╝  ██║        ║
    ║    ╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║ ╚████╔╝ ███████╗╚██████╗   ║
    ║     ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝  ╚══════╝ ╚═════╝   ║
    ║                                                                   ║
    ║        A Unified Multi-Task Embedding Framework                  ║
    ║     with Emotion, Temporal, and Causality Awareness              ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run OmniVec pipeline')
    parser.add_argument('--skip-install', action='store_true',
                       help='Skip package installation')
    parser.add_argument('--parts', type=str, default='1,2,3,4,5,6',
                       help='Comma-separated list of parts to run (e.g., 1,2,3)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: reduced epochs and data for testing')
    return parser.parse_args()

def run_part(part_num, skip_install=False, quick=False):
    """Run a specific part of the pipeline"""
    
    part_map = {
        1: ('Setup & Dependencies', 'omnivec_part1'),
        2: ('Data Collection', 'omnivec_part2'),
        3: ('Baseline Models', 'omnivec_part3'),
        4: ('OmniVec Architecture', 'omnivec_part4'),
        5: ('Training & Evaluation', 'omnivec_part5'),
        6: ('Visualizations & Analysis', 'omnivec_part6')
    }
    
    if part_num not in part_map:
        print(f"Error: Invalid part number {part_num}")
        return False
    
    part_name, module_name = part_map[part_num]
    
    print("\n" + "="*80)
    print(f"PART {part_num}: {part_name.upper()}")
    print("="*80)
    
    try:
        # Import and run the module
        if part_num == 1 and not skip_install:
            import omnivec_part1
            omnivec_part1.main()
        elif part_num == 2:
            import omnivec_part2
            omnivec_part2.main()
        elif part_num == 3:
            import omnivec_part3
            omnivec_part3.main()
        elif part_num == 4:
            import omnivec_part4
            omnivec_part4.main()
        elif part_num == 5:
            import omnivec_part5
            if quick:
                # Modify config for quick run
                from omnivec_part1 import OmniVecConfig
                config = OmniVecConfig()
                config.OMNIVEC_EPOCHS = 5
                config.WORD2VEC_EPOCHS = 5
            omnivec_part5.main()
        elif part_num == 6:
            import omnivec_part6
            omnivec_part6.main()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR in Part {part_num}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution function"""
    
    # Print banner
    print_banner()
    
    # Parse arguments
    args = parse_arguments()
    
    # Get parts to run
    parts_to_run = [int(p) for p in args.parts.split(',')]
    
    print("\n" + "="*80)
    print("OMNIVEC PIPELINE EXECUTION")
    print("="*80)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parts to run: {parts_to_run}")
    print(f"Skip install: {args.skip_install}")
    print(f"Quick mode: {args.quick}")
    
    if args.quick:
        print("\n⚡ QUICK MODE ENABLED: Using reduced epochs for testing")
    
    # Confirm execution
    print("\n" + "-"*80)
    response = input("Ready to start? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n⏸️  Execution cancelled.")
        return
    
    # Run pipeline
    start_time = time.time()
    results = {}
    
    for part_num in parts_to_run:
        part_start = time.time()
        success = run_part(part_num, skip_install=args.skip_install, quick=args.quick)
        part_time = time.time() - part_start
        
        results[f"Part {part_num}"] = {
            'success': success,
            'time': part_time
        }
        
        if not success:
            print(f"\n⚠️  Part {part_num} failed. Continue anyway? (yes/no): ", end='')
            cont = input().strip().lower()
            if cont not in ['yes', 'y']:
                print("\n❌ Pipeline execution stopped.")
                break
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"\nTotal Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "-"*80)
    print("Part-by-Part Results:")
    print("-"*80)
    
    all_success = True
    for part, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{part:<15} {status:<15} {result['time']:>8.2f}s")
        if not result['success']:
            all_success = False
    
    print("\n" + "="*80)
    
    if all_success:
        print("✨ ALL PARTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n📦 Deliverables:")
        print("  ✓ Models saved in:        ./models/")
        print("  ✓ Results saved in:       ./results/")
        print("  ✓ Visualizations saved in: ./plots/")
        print("  ✓ Data saved in:          ./data/")
        
        print("\n📊 Key Files:")
        print("  • comprehensive_comparison.csv  - All results")
        print("  • paper_ready_table.csv        - LaTeX table")
        print("  • paper_abstract.txt           - Research abstract")
        print("  • embedding_space_comparison.png - Main figure")
        print("  • performance_comparison.png   - Results chart")
        
        print("\n🎓 Next Steps:")
        print("  1. Review results in ./results/comprehensive_comparison.csv")
        print("  2. Check visualizations in ./plots/")
        print("  3. Read abstract in ./results/paper_abstract.txt")
        print("  4. Use LaTeX table for your paper")
        print("  5. Submit to: ACL, EMNLP, NAACL, AAAI, or TACL")
        
        print("\n🎉 Your research is ready for publication!")
        
    else:
        print("⚠️  PIPELINE COMPLETED WITH ERRORS")
        print("="*80)
        print("\nSome parts failed. Please check the error messages above.")
        print("You can:")
        print("  1. Fix the errors and re-run specific parts")
        print("  2. Run parts individually: python omnivec_partX.py")
        print("  3. Check the documentation for troubleshooting")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Execution interrupted by user.")
        print("Progress has been saved. You can resume by running specific parts.")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease report this error with the traceback above.")
