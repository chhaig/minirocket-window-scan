# Datasets and attribution

The repository contains original experimental result tables/JSON, protocol manifests and fitted classifier parameters. It does not mirror raw or converted provider recordings. Use the official sources and follow each provider's current terms and requested citations.

| Dataset | Official source | Preparation |
|---|---|---|
| HAPT | https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions | `download_inputs.py hapt`; acceleration axes only; official participant split, 1-based inclusive label intervals |
| WISDM v1.1 | https://www.cis.fordham.edu/wisdm/dataset.php | `download_inputs.py wisdm`, then `prepare_wisdm.py`; exact archive/raw hashes retained in `protocols/wisdm-source.json` |
| CWRU | https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data | `download_inputs.py cwru`, then `prepare_cwru.py`; 36 DE_time records and two split definitions in `protocols/cwru-manifest.json` |
| PAMAP2 | https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring | `download_uci.py`; Protocol files only; participants 101–108; channels/splits fixed in `protocols/subject-splits.json` |
| MHEALTH | https://archive.ics.uci.edu/dataset/319/mhealth+dataset | `download_uci.py`; nine acceleration channels; participant split 1–6/7/8–10 |
| Daphnet | https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait | `download_uci.py`; nine acceleration channels; participant split 1–6/7/8–10 |

Suggested source citations: Reyes-Ortiz et al., HAPT (UCI DOI 10.24432/C54G7M); Kwapisz et al., *Activity recognition using cell phone accelerometers*, 2011, DOI 10.1145/1964897.1964918; Reiss and Stricker, PAMAP2, 2012, DOI 10.1109/ISWC.2012.13; Banos et al., mHealthDroid, 2014, DOI 10.1007/978-3-319-13105-4_14; Bächlin et al., Daphnet, 2010, DOI 10.1109/TITB.2009.2036165; Case Western Reserve University Bearing Data Center.

PAMAP2/MHEALTH/Daphnet source archives were obtained from UCI in the original study; archive hashes and split statistics are retained in the prepared manifests. WISDM parsing retains zero and nonincreasing timestamps, splits at subject changes, gaps exceeding one second, and malformed observations, and discards segments shorter than one window. Labels are not relabelled or dropped to improve accuracy. See the included parser and manifests for the exact protocol.
