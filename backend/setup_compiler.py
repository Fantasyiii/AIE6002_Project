"""
Configure Python to use MinGW compiler instead of MSVC
"""
import os
import sys
import site

# Add MinGW to PATH
mingw_path = r"C:\Software\mingw64\bin"
if mingw_path not in os.environ["PATH"]:
    os.environ["PATH"] = mingw_path + os.pathsep + os.environ["PATH"]

# Set compiler environment variables
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

print("Compiler configuration:")
print(f"  CC = {os.environ.get('CC')}")
print(f"  CXX = {os.environ.get('CXX')}")
print(f"  PATH includes MinGW: {mingw_path in os.environ['PATH']}")
