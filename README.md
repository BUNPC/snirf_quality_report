# snirf_quality_report
Generates quality report for fNIRS acquisition data in SNIRF format

### Input
Takes path to SNIRF file located in BIDS folder structure

### Output
Saves quality_report.html and updated channles file in derivatives folder.

### Example
```
from snirf_quality_report.snirf_quality_report import snirf_quality_report

snirf_path = 'path_to_the_snirf_file_insie_a_BIDS_folder'
obj = snirf_quality_report()
obj.run_report(snirf_path)
 
```
