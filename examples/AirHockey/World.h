#ifndef _GameCtrl_World_h_
#define _GameCtrl_World_h_

namespace GameCtrl {
using namespace Upp;

class Object;
class ContactListener;

struct QueryCallback : b2QueryCallback
{
	QueryCallback(const b2Vec2& point)
	{
		this->point = point;
		fixture = NULL;
	}

	bool ReportFixture(b2Fixture* fixture)
	{
		b2Body* body = fixture->GetBody();
		if (body->GetType() == b2_dynamicBody)
		{
			bool inside = fixture->TestPoint(point);
			if (inside)
			{
				this->fixture = fixture;
				return false;
			}
		}

		return true;
	}

	b2Vec2 point;
	b2Fixture* fixture;
};

class WorldDebug : public b2DebugDraw {
	
protected:
	friend class World;
	
	Draw* w;
	Sizef sz;
	double cx, cy;
	double aspect;
	double zoom;
	double xoff, yoff;
public:
	WorldDebug();
	
	void SetOffset(double x, double y) {xoff = x; yoff = y;}
	
	void Init(Draw& d, Size s);
	Point conv(const b2Vec2& v);
	b2Vec2 conv(const Point& p);
	Color conv(const b2Color& c, double f = 255.0);
	void DrawPolygon(const b2Vec2* v, int vertexCount, const b2Color& color);
	void DrawSolidPolygon(const b2Vec2* v, int vertexCount, const b2Color& color);
	void DrawCircle(const b2Vec2& center, float32 radius, const b2Color& color);
	void DrawSolidCircle(const b2Vec2& center, float32 radius, const b2Vec2& axis, const b2Color& color);
	void DrawSegment(const b2Vec2& p1, const b2Vec2& p2, const b2Color& color);
	void DrawTransform(const b2Transform& xf);
	void DrawPoint(const b2Vec2& p0, float32 size, const b2Color& color);
	void DrawString(int x, int y, const char* string, ...);
	void DrawAABB(b2AABB* aabb, const b2Color& color);
	
};

class WorldDraw {
	
protected:
	friend class World;
	
	void* world;
	Draw* w;
	Sizef sz;
	double cx, cy;
	double aspect;
	double zoom;
	double xoff, yoff;
public:
	WorldDraw(void* world);
	
	double GetAspect() const {return aspect;}
	
	void SetOffset(double x, double y) {xoff = x; yoff = y;}
	
	void Init(Draw& d, Size s);
	Pointf ToScreen(const b2Vec2& v);
	Pointf ToScreen(const Pointf& p);
	Pointf ToScreen(double x, double y);
	b2Vec2 FromScreen(const Pointf& p);
	
	void DrawData();
	
};


class World : public Ctrl {
	
protected:
	friend class WorldDraw;
	
	b2World world;
	b2MouseJoint* mouseJoint;
	b2Vec2 mouseWorld;
	b2Body* groundBody;
	b2Vec2 gravity;
	b2BodyDef bodyDef;
	
	TimeStop ts;
	WorldDebug debugDraw;
	WorldDraw world_draw;
	int draw_mode;
	int tick_interval;
	bool sim_speed;
	bool dbg_showBoxes;
	bool draw_debug;
	bool running, stopped;
	
	VectorMap<unsigned, Object*> obj_list;
	
	enum {DRAW_DRAW, DRAW_PAINTER_NOT_AA, DRAW_PAINTER_AA, DRAW_PAINTER_AA_SUBPIX};
	
public:
	typedef World CLASSNAME;
	World();
	~World();
	
	void Start();
	void Stop();
	
	virtual void Paint(Draw& w);
	virtual void LeftDown(Point p0, dword keyflags);
	virtual void LeftUp(Point p0, dword keyflags);
	virtual void MouseMove(Point p, dword keyflags);
	virtual void MouseWheel(Point p, int zdelta, dword keyflags);
	virtual void Tick();
	void Ticking();
	
	void Render() {Refresh();}
	void SetSpeed(bool simulate_speed, int interval=100);
	void Add(Object& obj);
	void Remove(Object& obj);
	void SetGravityVert() {gravity.Set(0.0f, -10.0f); world.SetGravity(gravity);}
	void SetGravityZero() {gravity.Set(0.0, 0.0); world.SetGravity(gravity);}
	void SetGravity(double acc_x, double acc_y) {gravity.Set(acc_x, acc_y); world.SetGravity(gravity);}
	void SetContactListener(ContactListener& cl);
	void DebugDraw(bool b=true) {draw_debug = b;}
	
	
	// Box2D backend functions
	b2World& B2_GetWorld() {return world;}
	
	SpinLock lock;
	
};

}

#endif
