from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess


HERE = os.path.abspath(os.path.dirname(__file__))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exc:
            raise RuntimeError("CMake must be installed to build the extension") from exc

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_INCLUDE_DIRS={sys.prefix}/include/python{sys.version_info.major}.{sys.version_info.minor}",
            f"-DPython_LIBRARIES={sys.prefix}/lib/libpython{sys.version_info.major}.{sys.version_info.minor}.so",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg, "--", "-j"]
        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)


setup(
    name="arx5-interface-local",
    version="0.1.2",
    ext_modules=[CMakeExtension("arx5_interface", sourcedir=os.path.join(HERE, "python"))],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
