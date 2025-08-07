#!/usr/bin/env python3
"""
test_scraper.py - Test script for MLB_Scrape fixes

This script tests the updated MLB scraper to ensure all fixes work correctly
before integrating with the training pipeline.
"""

import sys
import os
from datetime import datetime

# Add the current directory to Python path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data_scraper import MLB_Scrape
    print("✅ Successfully imported MLB_Scrape")
except ImportError as e:
    print(f"❌ Failed to import MLB_Scrape: {e}")
    print("Make sure data_scraper.py is in the same directory")
    sys.exit(1)


def test_regex_fix():
    """Test the regex pattern fix in get_player_games_list"""
    print("\n🔍 Testing regex fix...")
    
    try:
        scraper = MLB_Scrape()
        
        # Test with a known pitcher (Gerrit Cole)
        games = scraper.get_player_games_list(
            player_id=592450,  # Gerrit Cole
            season=2024,
            start_date="2024-04-01",
            end_date="2024-04-30"
        )
        
        print(f"✅ Regex fix successful - found {len(games)} games for April 2024")
        return True
        
    except Exception as e:
        print(f"❌ Regex test failed: {e}")
        return False


def test_data_collection():
    """Test the main data collection functionality"""
    print("\n📊 Testing data collection...")
    
    try:
        scraper = MLB_Scrape()
        
        # Test with very limited data to avoid long wait times
        pitch_df, game_df, season_df = scraper.get_current_season_data_for_training(
            season_year=2024,
            max_players=3,  # Only test with 3 players
            test_mode=True
        )
        
        if pitch_df is not None:
            print(f"✅ Data collection successful!")
            print(f"   - Pitches collected: {len(pitch_df):,}")
            print(f"   - Players: {pitch_df['Name'].nunique()}")
            print(f"   - Game records: {len(game_df):,}")
            
            # Check required columns
            required_cols = ['Name', 'Season', 'game_date', 'is_swing', 'is_whiff']
            missing_cols = [col for col in required_cols if col not in pitch_df.columns]
            
            if missing_cols:
                print(f"⚠️ Missing required columns: {missing_cols}")
                return False
            else:
                print("✅ All required columns present")
                
            # Check data types
            print(f"✅ Data types look good:")
            print(f"   - game_date: {pitch_df['game_date'].dtype}")
            print(f"   - is_swing: {pitch_df['is_swing'].dtype}")
            
            return True
        else:
            print("❌ Data collection returned None")
            return False
            
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schedule_functionality():
    """Test basic schedule functionality"""
    print("\n📅 Testing schedule functionality...")
    
    try:
        scraper = MLB_Scrape()
        
        # Get a small schedule sample
        schedule = scraper.get_schedule(
            year_input=[2024],
            sport_id=[1], 
            game_type=['R']
        )
        
        if schedule is not None and len(schedule) > 0:
            # Convert to pandas if it's polars
            if hasattr(schedule, 'to_pandas'):
                schedule_df = schedule.to_pandas()
            else:
                schedule_df = schedule
                
            print(f"✅ Schedule test successful - found {len(schedule_df)} games")
            return True
        else:
            print("❌ Schedule test failed - no games found")
            return False
            
    except Exception as e:
        print(f"❌ Schedule test failed: {e}")
        return False


def test_integration_compatibility():
    """Test compatibility with training pipeline format"""
    print("\n🔗 Testing training pipeline compatibility...")
    
    try:
        scraper = MLB_Scrape()
        
        # Get minimal data for testing
        pitch_df, game_df, season_df = scraper.get_current_season_data_for_training(
            season_year=2024,
            max_players=2,
            test_mode=True
        )
        
        if pitch_df is None:
            print("❌ No data collected for compatibility test")
            return False
        
        # Test saving and loading (like training pipeline does)
        test_dir = "test_output"
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            pitch_df.to_csv(f'{test_dir}/test_pitch.csv', index=False)
            game_df.to_csv(f'{test_dir}/test_game.csv', index=False)
            season_df.to_csv(f'{test_dir}/test_season.csv', index=False)
            
            # Try to read them back
            import pandas as pd
            test_pitch = pd.read_csv(f'{test_dir}/test_pitch.csv')
            test_game = pd.read_csv(f'{test_dir}/test_game.csv')
            test_season = pd.read_csv(f'{test_dir}/test_season.csv')
            
            print("✅ File I/O compatibility test passed")
            
            # Clean up test files
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
            
            return True
            
        except Exception as e:
            print(f"❌ File I/O test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Integration compatibility test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 Starting MLB Scraper Tests")
    print("=" * 50)
    
    tests = [
        ("Regex Fix", test_regex_fix),
        ("Schedule Functionality", test_schedule_functionality),
        ("Data Collection", test_data_collection),
        ("Integration Compatibility", test_integration_compatibility),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Scraper is ready for integration.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # You can run individual tests or all tests
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        if test_name == "regex":
            test_regex_fix()
        elif test_name == "data":
            test_data_collection()
        elif test_name == "schedule":
            test_schedule_functionality()
        elif test_name == "integration":
            test_integration_compatibility()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: regex, data, schedule, integration")
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)