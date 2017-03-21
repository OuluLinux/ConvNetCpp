#ifndef _GameCtrl_Contact_h_
#define _GameCtrl_Contact_h_

namespace GameCtrl {
using namespace Upp;

class Contact {
	b2Contact* contact;
	
public:
	Contact();
	Contact(b2Contact* c);
	
	bool IsTouching() const {return contact->IsTouching();}
	void SetEnabled(bool b) {contact->SetEnabled(b);}
	Contact GetNext() {return contact->GetNext();}
	
	Object* GetObjectA() {return static_cast<Object*>(contact->GetFixtureA()->GetUserData());}
	Object* GetObjectB() {return static_cast<Object*>(contact->GetFixtureB()->GetUserData());}
	
	template <class T> T* GetA() {return dynamic_cast<T*>(GetObjectA());}
	template <class T> T* GetB() {return dynamic_cast<T*>(GetObjectB());}
	template <class T> T* Get() {T* a = GetA<T>(); if (a) return a; return GetB<T>();}
	
};

class ContactListener : public b2ContactListener {
	
protected:
	virtual void BeginContact(b2Contact* contact);
	virtual void EndContact(b2Contact* contact);
	
public:
	ContactListener();
		
	virtual void ContactBegin(Contact contact) {}
	virtual void ContactEnd(Contact contact) {}
	
};

}

#endif
