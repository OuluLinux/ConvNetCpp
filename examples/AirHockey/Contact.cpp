#include "AirHockey.h"

namespace GameCtrl {
using namespace Upp;

Contact::Contact() {
	
}

Contact::Contact(b2Contact* c) : contact(c) {
	
}




ContactListener::ContactListener() {
	
}

void ContactListener::BeginContact(b2Contact* contact) {
	ContactBegin(contact);
}

void ContactListener::EndContact(b2Contact* contact) {
	ContactEnd(contact);
}



}
