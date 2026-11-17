# Local data (never committed)

Obtain the restricted match bundle through the organizers. Keep it outside the repository,
or unpack it here. Prediction requires only:

```
data/
  tactical/
    BAR-ATH_tactical_atd.xml
    BAR-ATH_tactical_atd.txt
```

Set `MTAH_DATA_DIR` to the parent of `tactical/` to use another location. See the main
README for PowerShell and Linux commands. Videos, homographies and Smart Tags are not
read by the final model.

Training downloads go to ignored `data/external/dfl/`. Parsed tracking, features and
probability caches go to ignored `data/cache/`. Neither folder belongs in a public commit.

[Official data instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md)
