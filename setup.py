import os
import platform
import subprocess
import sys
from pprint import pprint
import pathlib
import shutil

from distutils.command.install_data import install_data
from distutils.command.install_headers import install_headers
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from setuptools.command.install_scripts import install_scripts

# Filename for the C extension module library
c_module_name = 'galario'

# Command line flags forwarded to CMake (for debug purpose)
cmake_cmd_args = []
for f in sys.argv:
    if f.startswith('-D'):
        cmake_cmd_args.append(f)

for f in cmake_cmd_args:
    sys.argv.remove(f)


def _get_env_variable(name, default='OFF'):
    if name not in os.environ.keys():
        return default
    return os.environ[name]


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='../..', sources=[], **kwa):
        Extension.__init__(self, name, sources=sources, **kwa)
        #self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)
        self.cmake_lists_dir = cmake_lists_dir


class CMakeBuild(build_ext):

    def build_extensions(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        for ext in self.extensions:
            cmake_args = [
                    '-DCMAKE_INSTALL_PREFIX=../../{}'.format(self.build_temp),
                    '-DGALARIO_CHECK_CUDA=0',
                    '-DPython_ADDITIONAL_VERSIONS={0:d}.{1:d}'.format(
                        sys.version_info[0], sys.version_info[1]),
            ]

            cmake_args += cmake_cmd_args

            pprint(cmake_args)

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            extension_path = "{}".format(self.build_lib)

            if not os.path.exists(extension_path):
                os.makedirs(extension_path)

            # Config and build the extension
            subprocess.check_call(["env"]+ext.extra_compile_args+\
                    ['cmake', ext.cmake_lists_dir] + cmake_args,
                    cwd=self.build_temp)
            subprocess.check_call(["env"]+ext.extra_compile_args+\
                    ['make'], cwd=self.build_temp)
            subprocess.check_call(["env"]+ext.extra_compile_args+\
                    ['make', 'install'], cwd=self.build_temp)

            # Copy files to the relevant location.

            bin_dir = self.build_temp
            self.distribution.bin_dir = bin_dir

            pyd_path = os.path.join(bin_dir, "lib", "python{0:d}.{1:d}".format(
                sys.version_info[0], sys.version_info[1]), "site-packages",
                "galario")

            shutil.move(pyd_path, extension_path)

class InstallCMakeHeaders(install_headers):
    def run(self):
        print(self.install_dir)

        headers = ["{0:s}/include/{1:s}".format(self.distribution.bin_dir, 
            header) for header in ["galario.h","galario_defs.h","galario_py.h"]]

        for header in headers:
            dst = os.path.join(self.install_dir, os.path.dirname(header.
                split("/")[-1]))
            self.mkpath(dst)
            (out, _) = self.copy_file(header, dst)
            self.outfiles.append(out)

class InstallCMakeLibsData(install_data):
    def run(self):
        print(self.install_dir)

        if sys.platform == 'darwin':
            fileext = ".dylib"
        else:
            fileext = ".so"

        libs = ["{0:s}/lib/{1:s}".format(self.distribution.bin_dir, 
            lib) for lib in ["libgalario"+fileext,"libgalario_single"+fileext]]

        for lib in libs:
            dst = os.path.join(self.install_dir, "lib", os.path.dirname(lib.
                split("/")[-1]))
            self.mkpath(dst)
            (out, _) = self.copy_file(lib, dst)
            self.outfiles.append(out)

class InstallCMakeLibs(install_lib):
    def run(self):
        super().run()

        self.distribution.run_command("install_data")
        self.distribution.run_command("install_headers")

# Check which set of extra compile args are needed, based on OS.

extra_compile_args = []

if sys.prefix == 'darwin':
    extra_compile_args += ['LDFLAGS="-Wl,-rpath='+sys.base_prefix+'/lib"']
else:
    extra_compile_args += ['LDFLAGS="-Wl,-rpath,'+sys.base_prefix+'/lib"']

# The following line is parsed by Sphinx
version = '1.2.2'

setup(name='galario',
      version=version,
      description='Gpu Accelerated Library for Analysing Radio Interferometer Observations',
      author='Marco Tazzari',
      url='https://mtazzari.github.io/galario',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      install_requires=['numpy','pytest','cython'],
      ext_modules=[CMakeExtension(c_module_name, 
          extra_compile_args=extra_compile_args)],
      cmdclass={
          'build_ext': CMakeBuild,
          'install_headers': InstallCMakeHeaders,
          'install_data': InstallCMakeLibsData,
          'install_lib': InstallCMakeLibs},
      zip_safe=False,
      )
