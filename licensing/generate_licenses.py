
import os as os
import shutil
from glob import glob
import re as regex

whereamI = os.getcwd()
if whereamI[-11:] != "MIW_AutoFit" :
    print(f"You need to be in the autofit directory to run this. You are in {whereamI}")
os.chdir('..')
whereamI = os.getcwd()
if whereamI[-11:] != "MIW_AutoFit" :
    print(f"You need to be in the autofit directory to run this. You are in {whereamI}")
    raise SystemExit
cwd = os.getcwd()

# copy the AutoFit license to the dist directory
shutil.copy(f"{cwd}/LICENSE" , f"{cwd}/autofit/dist/LICENSE")

# loop over packages in the dist folder to find the package names
package_paths = glob(f"{cwd}/autofit/dist/MIW_autofit/*/", recursive=True)
package_names = [ regex.split(f"\\\\", path)[-2] for path in package_paths]
for name in package_names[:] :
    if name[-9:] == "dist-info" :
        continue
    print(name)
    f = open(f"{cwd}/licensing/subpackage_licenses/{name}.LICENSE",'a')
    f.close()
# get the license for each package from the venv directories
