#ifndef _GameCtrl_Object_h_
#define _GameCtrl_Object_h_

// Set few box2d classes as moveable
NAMESPACE_UPP
NTL_MOVEABLE(b2Vec2);
END_UPP_NAMESPACE

namespace GameCtrl {
using namespace Upp;

class Object {
	
protected:
	friend class World;
	
	b2FixtureDef fd;
	b2BodyDef bd;
	World* world;
	b2Fixture* fix;
	b2Body* body;
	
	void SetWorld(World* w) {world = w;}
	
public:
	Object();
	
	virtual void Paint(WorldDraw& wdraw, Draw& draw) {}
	
	// Functions, which works before Create
	void SetTypeDynamic() {bd.type = b2_dynamicBody;}
	void SetTypeStatic() {bd.type = b2_staticBody;}
	void SetPosition(double x, double y) {bd.position.Set(x, y);}
	void SetFriction(double f) {fd.friction = f;}
	void SetRestitution(double f) {fd.restitution = f;}
	void SetDensity(double f) {fd.density = f;}
	void SetSensor(bool b=true) {fd.isSensor = b;}
	void FilterAllCollision() {fd.filter.maskBits = 0;}
	
	virtual void Create();
	
	
	// Functions, which works after Create
	void SetCategory(int b16_pos, bool value);
	void SetCollisionFilter(int b16_pos, bool value);
	void SetSpeed(const Pointf& s) {body->SetLinearVelocity(b2Vec2(s.x, s.y));}
	void SetSpeed(double x, double y) {body->SetLinearVelocity(b2Vec2(x, y));}
	
	void ApplyForceToCenter(const Pointf& f) {body->ApplyForce(b2Vec2(f.x, f.y), body->GetLocalCenter());}
	
	bool IsColliding(const Object& o) const;
	bool IsDynamic() const {return body ? body->GetType() == b2_dynamicBody : bd.type == b2_dynamicBody;}
	
	Pointf GetPosition() const {const b2Vec2& pos = body->GetPosition(); return Pointf(pos.x, pos.y);}
	Pointf GetSpeed() const {b2Vec2 vel = body->GetLinearVelocity(); return Pointf(vel.x, vel.y);}
	double GetRotation() const {return body->GetAngularVelocity();}
	double GetAngle() const {return body->GetAngle();}
	
	
};


class Circle : public Object {
	b2CircleShape shape;
	
public:
	Circle();
	
	void PaintCircle(WorldDraw& wdraw, Draw& draw, Color fill_color, Color border_color);
	
	double GetRadius() const {return shape.m_radius;}
	
	Circle& SetRadius(double d) {shape.m_radius = d; return *this;}
};


class Polygon : public Object {
	Vector<b2Vec2> vertices;
	b2PolygonShape shape;
	
public:
	Polygon();
	
	void PaintPolyline(WorldDraw& wdraw, Draw& draw, Color color=Black(), int width=1);
	void PaintPolygon(WorldDraw& wdraw, Draw& draw, Color fill_color, int width=1, Color border_color=Black());
	
	Polygon& AddVertex(double x, double y) {vertices.Add().Set(x, y); return *this;}
	Polygon& AddVertex(Pointf pt) {vertices.Add().Set(pt.x, pt.y); return *this;}
	Polygon& operator << (Pointf pt) {vertices.Add().Set(pt.x, pt.y); return *this;}
	
	virtual void Create();
	
	int GetCount() const {return vertices.GetCount();}
	Pointf operator[](int i){b2Vec2& b = vertices[i]; return Pointf(b.x, b.y);}
	
};


}

#endif
