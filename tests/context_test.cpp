#include <tender/context.hpp>

#include <gtest/gtest.h>

using namespace tender;

// ---- alloc_index_id ----------------------------------------------------

TEST(Context, AllocIndexIdStartsAtZero)
{
    Context ctx;
    EXPECT_EQ(ctx.alloc_index_id(), 0);
}

TEST(Context, AllocIndexIdIsMonotone)
{
    Context ctx;
    int a = ctx.alloc_index_id();
    int b = ctx.alloc_index_id();
    int c = ctx.alloc_index_id();
    EXPECT_LT(a, b);
    EXPECT_LT(b, c);
}

// ---- make<T> -----------------------------------------------------------

TEST(Context, MakeReturnsNonNull)
{
    Context ctx;
    auto* p = ctx.make<int>(42);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(*p, 42);
}

TEST(Context, MakeOwnsObject)
{
    Context ctx;
    auto* p = ctx.make<std::string>("hello");
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(*p, "hello");
}

TEST(Context, MultipleAllocationsAreDistinct)
{
    Context ctx;
    auto* a = ctx.make<int>(1);
    auto* b = ctx.make<int>(2);
    EXPECT_NE(a, b);
    EXPECT_EQ(*a, 1);
    EXPECT_EQ(*b, 2);
}

// ---- new_context -------------------------------------------------------

TEST(Context, NewContextSharesIdFactory)
{
    Context parent;
    parent.alloc_index_id(); // 0
    parent.alloc_index_id(); // 1

    Context child = parent.new_context();
    EXPECT_EQ(child.alloc_index_id(), 2); // continues from parent
}

TEST(Context, IdsContinueAcrossMultipleChildren)
{
    Context parent;
    Context child1 = parent.new_context();
    child1.alloc_index_id(); // 0
    child1.alloc_index_id(); // 1

    Context child2 = parent.new_context();
    EXPECT_EQ(child2.alloc_index_id(), 2); // continues, not reset
}

TEST(Context, ParentAndChildShareFactory)
{
    Context parent;
    Context child = parent.new_context();

    int a = parent.alloc_index_id(); // 0
    int b = child.alloc_index_id();  // 1 — shared factory
    int c = parent.alloc_index_id(); // 2

    EXPECT_EQ(a, 0);
    EXPECT_EQ(b, 1);
    EXPECT_EQ(c, 2);
}

TEST(Context, ChildHasIndependentResourceList)
{
    Context parent;
    auto* p = parent.make<int>(10);

    Context child = parent.new_context();
    auto* q = child.make<int>(20);

    // Both allocations are valid and independent.
    EXPECT_EQ(*p, 10);
    EXPECT_EQ(*q, 20);
    EXPECT_NE(static_cast<void*>(p), static_cast<void*>(q));
}

// ---- Move semantics ----------------------------------------------------

TEST(Context, MoveConstructed)
{
    Context ctx;
    ctx.alloc_index_id(); // 0

    Context moved = std::move(ctx);
    EXPECT_EQ(moved.alloc_index_id(), 1); // factory carried over
}

TEST(Context, MoveAssigned)
{
    Context src;
    src.alloc_index_id(); // 0

    Context dst;
    dst = std::move(src);
    EXPECT_EQ(dst.alloc_index_id(), 1);
}

// ---- Static checks -----------------------------------------------------

TEST(Context, CopyAssignmentIsDeleted)
{
    static_assert(!std::is_copy_assignable_v<Context>);
}
