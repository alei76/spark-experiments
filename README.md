Spark Experiments
=================

Spark 2.0.x experiments.

Data
----

Some sample data is in src/main/data

newsgroups:
Newsgroup data converted to sequence file format

ntsb:
NTSB data extracted from PDF files in an odd CSV format

Source
------

See:
drew.ml.JavaClassificationExample

Run
---

mvn exec:java -Dexec.mainClass=drew.ml.JavaClassificationExample
