# QWEN - Simple Guide for Ultimate++ Project

**Project**: ConvNetCpp
**Type**: Ultimate++ (U++) Package
**For**: Qwen AI - Simple, clear instructions

---

## STEP 1: Read AGENTS.md First!

**IMPORTANT**: Read the AGENTS.md file in this directory first!
It has all the detailed rules for U++ packages.

---

## STEP 2: Most Important Rules

### Rule #1: Every .cpp File Must Start With This

```cpp
#include "ConvNetCpp.h"  // ALWAYS FIRST!
```

**Why?** U++ uses BLITZ build system. This pattern is required!

**WRONG Examples:**
```cpp
// DON'T do this:
#include <vector>           // Wrong! System headers later!
#include "SomeOtherFile.h"  // Wrong! Only main header!
```

### Rule #2: Don't Use std:: Types

Use U++ types instead:

| DON'T Use | DO Use |
|-----------|--------|
| std::string | String |
| std::vector<T> | Vector<T> or Array<T> |
| std::map<K,V> | VectorMap<K,V> |
| std::unique_ptr<T> | One<T> |
| std::optional<T> | Optional<T> |
| .find() | .Find() |
| .substr() | .Mid() |
| std::cout | Upp::Cout() |

### Rule #3: No #include in Non-Main Headers

```cpp
// In non-main headers (NOT ConvNetCpp.h):
// DON'T add #include statements!
// Only the main header includes other files!

// You can add as COMMENT for reference:
// #include <SomeOtherPackage/SomeOtherPackage.h>
```

### Rule #4: Use RAII Containers

```cpp
// WRONG - Don't use smart pointers in containers
Vector<Ptr<Node>> nodes;

// RIGHT - Use proper RAII containers
Array<Node> nodes;
```

---

## STEP 3: Building This Project

### Using TheIDE
1. Open TheIDE (Ultimate++ IDE)
2. Open the .upp file
3. Press F5 to build

### Using umk (command line)
```bash
umk ConvNetCpp GCC -r
```

---

## STEP 4: Common Mistakes to Avoid

### Mistake 1: Wrong Include Order
```cpp
// WRONG
#include <iostream>
#include "ConvNetCpp.h"

// RIGHT
#include "ConvNetCpp.h"
// other includes can go after if really needed
```

### Mistake 2: Using std:: Instead of U++
```cpp
// WRONG
std::string name = "test";
int pos = name.find("e");

// RIGHT
String name = "test";
int pos = name.Find("e");
```

### Mistake 3: Including Headers in Non-Main Headers
```cpp
// In SomeClass.h (NOT the main header):
// WRONG
#include <OtherPackage/OtherPackage.h>

// RIGHT - just use forward declaration or comment
// #include <OtherPackage/OtherPackage.h>  // Comment only!
```

---

## STEP 5: Quick Reference

### Find Operations
```cpp
// std::string way (DON'T use):
size_t pos = str.find("text");
if (pos != std::string::npos) { ... }

// U++ way (DO use):
int pos = str.Find("text");
if (pos != -1) { ... }
```

### Container Iteration
```cpp
// With VectorMap:
VectorMap<String, String> map;
for (const auto pair : ~map) {  // Note the ~ operator
    String key = pair.key;
    String value = pair.value;
}
```

### String Operations
```cpp
String str = "hello world";
String sub = str.Mid(0, 5);        // substring
int pos = str.Find("world");       // find
bool ends = str.EndsWith("ld");    // endswith check
```

---

## STEP 6: Where to Get Help

1. **Read AGENTS.md** in this directory
2. **Read UPP_CONVENTION.md** at /common/active/sblo/Dev/Manager/UPP_CONVENTION.md
3. **Look at other U++ packages** for examples
4. **Check .tpp files** for API documentation

---

## STEP 7: Project Workflow (IMPORTANT!)

### Task Tracking
- Keep tasks in **TASKS.md** with sections: TODO, IN_PROGRESS, DONE
- Tasks have phase numbers (1.1, 1.2, 2.1, etc.)
- **ALWAYS update TASKS.md** after completing a task!

### Git Commits
**VERY IMPORTANT - Read this!**

Before committing:
```bash
# 1. Build if script exists
[ -f build.sh ] && ./build.sh

# 2. Test if script exists
[ -f test.sh ] && ./test.sh

# 3. Update TASKS.md (move task to DONE)

# 4. Commit
git add .
git commit -m "Task X.Y: What you did

- Detail 1
- Detail 2

Phase: N (Phase Name)
Status: DONE"

# 5. ALWAYS PUSH!
git push
```

**Remember**: ALWAYS compile and test before commit!

### Documentation
- Use `docs/` or `doc/` for documentation
- Write `.puml` files for diagrams
- Generate PNG: `plantuml docs/*.puml`
- Include images in markdown files

### Roadmap
- Use `roadmap/` for version planning (v1.0.0.md, v1.1.0.md)
- Plan features for each version

### Pseudocode
- Use `pseudocode/` for algorithm design
- Write pseudocode BEFORE implementation
- Combine with UML diagrams

### For Qwen AI: Complex Planning
If you encounter difficult planning, ask user:
> "Should I use OpenAI API for ChatGPT-Codex planning?
> 1 = Use ChatGPT (costs API credits, better planning)
> 2 = Continue with Qwen
> 3 = Ask me questions"

---

## Quick Checklist

Before you write code:
- [ ] Did I read AGENTS.md?
- [ ] Do I start .cpp files with `#include "ConvNetCpp.h"`?
- [ ] Am I using U++ types (String, Vector, Array) not std:: types?
- [ ] Am I NOT adding #include to non-main headers?
- [ ] Am I using RAII containers (Array, not Ptr)?

**If NO to any: STOP and fix it!**

---

## Remember

**Three most important things:**

1. **Every .cpp starts with**: `#include "ConvNetCpp.h"`
2. **Use U++ types** not std:: types
3. **Only main header** includes other files

---

## Git Commit Guidelines

- Do not include `Co-authored-by: Qwen-Coder <qwen-coder@alibabacloud.com>` in git commits
- Qwen can be mentioned in documentation files like README.md, but not in commit messages
- Follow the project's standard commit message format without adding co-author attributions for Qwen

---

**Read AGENTS.md for complete details!**
