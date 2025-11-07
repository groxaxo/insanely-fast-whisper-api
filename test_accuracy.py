#!/usr/bin/env python3
"""
Test script to evaluate Whisper API accuracy with various audio files.
Tests language auto-detection and transcription quality.
"""

import requests
import json
import time
from pathlib import Path

API_URL = "http://localhost:8002/audio/transcriptions"

# Test files
TEST_FILES = [
    "/home/op/neutts-air/test.wav",
    "/home/op/ComfyUI/input/vozespanola.mp3",
    "/home/op/ComfyUI/input/andrewhubs.mp3",
    "/home/op/ComfyUI/input/lucho.mp3",
    "/home/op/happiness_book_output/chapter_01.wav",
]

def test_transcription(file_path, language=None):
    """Test transcription with a specific file."""
    if not Path(file_path).exists():
        print(f"  ⚠️  File not found: {file_path}")
        return None
    
    print(f"\n{'='*80}")
    print(f"Testing: {Path(file_path).name}")
    print(f"{'='*80}")
    
    # Get file size
    file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")
    
    # Prepare request
    files = {"file": open(file_path, "rb")}
    data = {"model": "whisper-large-v3"}
    
    if language:
        data["language"] = language
        print(f"Language specified: {language}")
    else:
        print("Language: Auto-detect")
    
    # Make request
    print("Transcribing...")
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, files=files, data=data, timeout=300)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("text", "")
            
            print(f"\n✅ Success! (took {elapsed:.2f}s)")
            print(f"\nTranscription ({len(text)} characters):")
            print("-" * 80)
            # Print first 500 characters
            if len(text) > 500:
                print(text[:500] + "...")
                print(f"\n[... {len(text) - 500} more characters ...]")
            else:
                print(text)
            print("-" * 80)
            
            # Calculate speed
            if file_size > 0:
                speed = file_size / elapsed
                print(f"\nProcessing speed: {speed:.2f} MB/s")
            
            return {
                "file": Path(file_path).name,
                "success": True,
                "text": text,
                "duration": elapsed,
                "size_mb": file_size,
                "char_count": len(text)
            }
        else:
            print(f"\n❌ Error {response.status_code}: {response.text}")
            return {
                "file": Path(file_path).name,
                "success": False,
                "error": response.text,
                "duration": elapsed
            }
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Exception: {str(e)}")
        return {
            "file": Path(file_path).name,
            "success": False,
            "error": str(e),
            "duration": elapsed
        }
    finally:
        files["file"].close()


def main():
    print("\n" + "="*80)
    print("WHISPER API ACCURACY TEST")
    print("="*80)
    print(f"\nAPI Endpoint: {API_URL}")
    print(f"Model: whisper-large-v3")
    print(f"Testing {len(TEST_FILES)} files")
    
    results = []
    
    for file_path in TEST_FILES:
        result = test_transcription(file_path)
        if result:
            results.append(result)
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful
    
    print(f"\nTotal tests: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    if successful > 0:
        avg_duration = sum(r["duration"] for r in results if r.get("success")) / successful
        total_chars = sum(r.get("char_count", 0) for r in results if r.get("success"))
        print(f"\nAverage processing time: {avg_duration:.2f}s")
        print(f"Total characters transcribed: {total_chars:,}")
    
    # Save detailed results
    output_file = "/home/op/CascadeProjects/insanely-fast-whisper-api/test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if failed > 0:
        print("\n⚠️  Some tests failed. Check:")
        print("  1. API is running on port 8002")
        print("  2. GPU has enough memory")
        print("  3. Audio files are valid")
    
    print("\nTo improve accuracy:")
    print("  • Ensure audio quality is good (clear speech, minimal noise)")
    print("  • For better results with specific languages, specify the language code")
    print("  • Consider using whisper-large-v3-turbo for faster processing")
    print("  • Check if Flash Attention 2 is properly configured")


if __name__ == "__main__":
    main()
