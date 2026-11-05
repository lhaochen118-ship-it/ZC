import json, sys
T = {"q_3fb7bbc8ec8a": 167, "q_32b653051a42": 59, "q_473593cb045c": ["Xiaomi Redmi Note 14", "iPhone 16", "iPhone 16 Pro Max"],
 "q_7826ef06d11c": ["iPhone 16 Pro Max", "iPhone 15", "Xiaomi Redmi Note 14"], "q_4b82c6814a0d": 78.49, "q_3ac94b30991c": 77.26,
 "q_792e1070baaf": 27.82, "q_d08785ea5577": 11025908, "q_8e50e264d157": 9563422, "q_4c24acd737a8": 1706335000, "q_763243ad81fd": 1546763000,
 "q_f30748b72898": 3702376, "q_e0ac213834f6": 4259801, "q_94f634ace6b1": 4325306, "q_c4da058cd902": "iPhone 16 128GB Đen", "q_e4fc0d18d409": 11,
 "q_b6abccb6b3cb": 86, "q_4162fa76a222": 9, "q_31dad3219917": 3116557000,
 "q_fbf739189e42": 4549169, "q_db00aee1a23d": 15, "q_7f91e434b0fc": 33, "q_c3158f13ca1c": 200, "q_8ce765944937": 14,
 "q_12712b3cc328": 6, "q_d622b78573c8": 7, "q_83e204622a9f": 6825000, "q_97a88c5a5568": 26, "q_a652f5fcb0fe": 34, "q_df258c67efec": 19, "q_3176702f9d3e": "Kính cường lực 9D",
 "q_10831a6f3a2b": 152, "q_1adddf599009": 145, "q_7305db18bf51": 19, "q_9eddae22a244": 164, "q_a18fefbb07ad": 21, "q_af96aaf1f4e2": 35}
for p in sys.argv[1:]:
    d = {r["id"]: r for r in map(json.loads, open(p))}; miss = [q for q, t in T.items() if d[q].get("answer", d[q]["type"]) != t]
    print(p.split("/")[-2], len(T) - len(miss), "/", len(T), "miss:", [(q[-6:], d[q].get("answer", d[q]["type"])) for q in miss])
