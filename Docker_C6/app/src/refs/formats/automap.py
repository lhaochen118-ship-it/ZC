# deterministic mapping of any export (src/automap.py): README column descriptions + name lexicon + value profiles ->
# table kinds -> canonical tables; conventions (status in notes, pseudo-lines, toppings, short names, unit rules ...).
import automap
C, NOTES, _rep = automap.build(BIZ)
for _t in _rep.get("used_tables", []): T[_t]                         # tables read count as used (completeness check)
