"""Gets the licensing info """

import os as os
import shutil
from glob import glob
import re as regex

curr_loc = os.getcwd()
if curr_loc[-11:] != "MIW_AutoFit":
    print(f"You need to be in the autofit directory to run this. You are in {curr_loc}")
os.chdir("..")
curr_loc = os.getcwd()
if curr_loc[-11:] != "MIW_AutoFit":
    print(f"You need to be in the autofit directory to run this. You are in {curr_loc}")
    raise SystemExit
cwd = os.getcwd()

# copy the AutoFit license to the dist directory
shutil.copy(f"{cwd}/LICENSE", f"{cwd}/autofit/dist/LICENSE")

# loop over packages in the dist folder to find the package names
package_paths = glob(f"{cwd}/autofit/dist/MIW_autofit/*/", recursive=True)
package_names = [regex.split(f"\\\\", path)[-2] for path in package_paths]
for name in package_names[:]:
    if name[-9:] == "dist-info":
        continue
    print(name)
    f = open(f"{cwd}/licensing/subpackage_licenses/{name}.LICENSE", "a")
    f.close()
# get the license for each package from the venv directories
