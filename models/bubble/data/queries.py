create_sequence_table: str = (
    'CREATE TABLE "sequence" ('
    '	"id"     INTEGER NOT NULL UNIQUE,'
    '	"length" INTEGER NOT NULL,'
    '	PRIMARY KEY("id" AUTOINCREMENT)'
    ")"
)

create_mesh_table: str = (
    'CREATE TABLE "mesh" ('
    '	"id"	INTEGER NOT NULL UNIQUE,'
    '	"faces"	BLOB NOT NULL,'
    '	"connect"	BLOB NOT NULL,'
    '	PRIMARY KEY("id" AUTOINCREMENT)'
    ")"
)

create_geometry_table: str = (
    'CREATE TABLE "geometry" ('
    '	"id"	INTEGER NOT NULL UNIQUE,'
    '	"mesh_id"	INTEGER NOT NULL,'
    '	"positions"	BLOB NOT NULL,'
    '	FOREIGN KEY("mesh_id") REFERENCES "mesh"("id"),'
    '	PRIMARY KEY("id" AUTOINCREMENT)'
    ")"
)

create_bubble_table: str = (
    'CREATE TABLE "bubble" ('
    '	"id"	INTEGER NOT NULL UNIQUE,'
    '	"sequence_id"	INTEGER NOT NULL,'
    '	"previous"	INTEGER NOT NULL,'
    '	"current"	INTEGER NOT NULL,'
    '	"next"	INTEGER NOT NULL,'
    '	FOREIGN KEY("previous") REFERENCES "geometry"("id"),'
    '	FOREIGN KEY("current") REFERENCES "geometry"("id"),'
    '	FOREIGN KEY("next") REFERENCES "geometry"("id"),'
    '	FOREIGN KEY("sequence_id") REFERENCES "sequence"("id"),'
    '	PRIMARY KEY("id" AUTOINCREMENT)'
    ")"
)

select_bubble_by_id: str = (
    "SELECT mesh.faces, mesh.connect, prev.positions as prev,"
    " cur.positions as cur, next.positions as next"
    " FROM bubble"
    " LEFT JOIN geometry prev"
    " ON bubble.previous = prev.id"
    " LEFT JOIN geometry cur"
    " ON bubble.current = cur.id"
    " LEFT JOIN geometry next"
    " ON bubble.next = next.id"
    " LEFT JOIN mesh"
    " ON prev.mesh_id = mesh.id"
    " WHERE bubble.id = ?"
)
