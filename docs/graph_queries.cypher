MATCH (u:User {user_id: $user_id})-[:FOLLOWS]->(friend)-[:WATCHED]->(m:Movie)
RETURN DISTINCT m.title;

MATCH (u:User {user_id: $user_id})-[:WATCHED]->(m:Movie)<-[:ACTED_IN]-(a:Actor)
RETURN DISTINCT a.name;

MATCH (u:User {user_id: $user_id})-[:FOLLOWS]->(:User)-[:FOLLOWS]->(f2),
      (f2)-[:WATCHED]->(m:Movie)
RETURN DISTINCT f2.user_id, m.title;
