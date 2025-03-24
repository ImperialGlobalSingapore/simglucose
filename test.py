output = [dist.project_name.replace("Python", "") for dist in __import__('pkg_resources').working_set]
import sys
print(sys.executable)
print(sorted(output))
print('----'*20)
from pip._internal.operations.freeze import freeze

print(sorted(line.split('==')[0].split("@")[0] for line in freeze()))