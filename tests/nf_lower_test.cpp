#include <tender/nf_lower.hpp>

#include <tender/derivation.hpp>  // steps::canonicalize (differential harness)
#include <tender/index_space.hpp> // space_3d

#include <gtest/gtest.h>

using namespace tender;
using namespace tender::nf;

namespace
{

auto atom(Context& ctx, std::string_view name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 1);
}

// An abstract tensor object of the given rank.
auto atomr(Context& ctx, std::string_view name, int rank) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, rank);
}

// Assert the flattened layer equals the given (sign, body) pairs, in order.
void expect_terms(
    std::vector<SignedExpr> const& got,
    std::vector<std::pair<int, Expr const*>> const& want)
{
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i)
    {
        EXPECT_EQ(got[i].sign, want[i].first) << "term " << i;
        EXPECT_EQ(got[i].body, want[i].second) << "term " << i;
    }
}

} // namespace

// ---- additive flatten --------------------------------------------------

TEST(AdditiveFlatten, SingleLeaf)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    expect_terms(additive_flatten(a), {{+1, a}});
}

TEST(AdditiveFlatten, Sum)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    expect_terms(additive_flatten(make_sum(ctx, a, b)), {{+1, a}, {+1, b}});
}

TEST(AdditiveFlatten, Difference)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    expect_terms(
        additive_flatten(make_difference(ctx, a, b)), {{+1, a}, {-1, b}});
}

TEST(AdditiveFlatten, Negate)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    expect_terms(additive_flatten(make_negate(ctx, a)), {{-1, a}});
}

TEST(AdditiveFlatten, DoubleNegateCancels)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    expect_terms(
        additive_flatten(make_negate(ctx, make_negate(ctx, a))), {{+1, a}});
}

TEST(AdditiveFlatten, NestedSignPropagation)
{
    // (a + b) - (c - d)  ->  +a +b -c +d
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* d = atom(ctx, "d");
    auto const* e =
        make_difference(ctx, make_sum(ctx, a, b), make_difference(ctx, c, d));
    expect_terms(additive_flatten(e), {{+1, a}, {+1, b}, {-1, c}, {+1, d}});
}

TEST(AdditiveFlatten, SubtractedSumFlipsBoth)
{
    // a - (b + c)  ->  +a -b -c
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* e = make_difference(ctx, a, make_sum(ctx, b, c));
    expect_terms(additive_flatten(e), {{+1, a}, {-1, b}, {-1, c}});
}

TEST(AdditiveFlatten, ProductIsOpaqueLeaf)
{
    // (a ⊗ b) + c  ->  the product is one leaf, NOT split.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* prod = make_tensor_product(ctx, a, b);
    expect_terms(
        additive_flatten(make_sum(ctx, prod, c)), {{+1, prod}, {+1, c}});
}

TEST(AdditiveFlatten, NoDistributionOverInnerSum)
{
    // (a + b) ⊗ c  ->  one leaf (the product); the inner sum is untouched.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* prod = make_tensor_product(ctx, make_sum(ctx, a, b), c);
    expect_terms(additive_flatten(prod), {{+1, prod}});
}

// ---- multiplicative flatten --------------------------------------------

namespace
{
auto flat(int sign, Expr const* body) -> ProductParts
{
    return multiplicative_flatten(SignedExpr{sign, body});
}
} // namespace

TEST(MultiplicativeFlatten, BareAtomCarriesSign)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto pos = flat(+1, a);
    EXPECT_EQ(pos.coeff, Rational{1});
    ASSERT_EQ(pos.factors.size(), 1u);
    EXPECT_EQ(pos.factors[0], a);

    auto neg = flat(-1, a);
    EXPECT_EQ(neg.coeff, Rational{-1});
    EXPECT_EQ(neg.factors, (std::vector<Expr const*>{a}));
}

TEST(MultiplicativeFlatten, ScalarLiteralFolds)
{
    Context ctx;
    auto const* three = make_scalar(ctx, Rational{3});
    auto pp = flat(-1, three);
    EXPECT_EQ(pp.coeff, Rational{-3});
    EXPECT_TRUE(pp.factors.empty());
}

TEST(MultiplicativeFlatten, ProductChainPreservesOrder)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* prod =
        make_tensor_product(ctx, make_tensor_product(ctx, a, b), c);
    auto pp = flat(+1, prod);
    EXPECT_EQ(pp.coeff, Rational{1});
    EXPECT_EQ(pp.factors, (std::vector<Expr const*>{a, b, c}));
}

TEST(MultiplicativeFlatten, NumericFactorsMultiplyIntoCoeff)
{
    // 2 ⊗ (3 ⊗ a)  ->  coeff 6, factor a.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* two = make_scalar(ctx, Rational{2});
    auto const* three = make_scalar(ctx, Rational{3});
    auto const* e =
        make_tensor_product(ctx, two, make_tensor_product(ctx, three, a));
    auto pp = flat(+1, e);
    EXPECT_EQ(pp.coeff, Rational{6});
    EXPECT_EQ(pp.factors, (std::vector<Expr const*>{a}));
}

TEST(MultiplicativeFlatten, NestedNegateFoldsSign)
{
    // a ⊗ (-b)  ->  coeff -1, factors [a, b].
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* e = make_tensor_product(ctx, a, make_negate(ctx, b));
    auto pp = flat(+1, e);
    EXPECT_EQ(pp.coeff, Rational{-1});
    EXPECT_EQ(pp.factors, (std::vector<Expr const*>{a, b}));
}

TEST(MultiplicativeFlatten, NumericDivisionFolds)
{
    // a / 2  ->  coeff 1/2, factor a.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* e = make_scalar_div(ctx, a, make_scalar(ctx, Rational{2}));
    auto pp = flat(+1, e);
    EXPECT_EQ(pp.coeff, (Rational{1, 2}));
    EXPECT_EQ(pp.factors, (std::vector<Expr const*>{a}));
}

TEST(MultiplicativeFlatten, NonNumericDivisionStaysOpaque)
{
    // a / (b·c)  ->  coeff 1, one opaque ScalarDiv factor (no reciprocal yet).
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* div = make_scalar_div(ctx, a, make_dot(ctx, b, c));
    auto pp = flat(+1, div);
    EXPECT_EQ(pp.coeff, Rational{1});
    ASSERT_EQ(pp.factors.size(), 1u);
    EXPECT_EQ(pp.factors[0], div);
}

TEST(MultiplicativeFlatten, ContractionIsOneFactorNotFlattened)
{
    // a ⊗ (b·c)  ->  two factors [a, (b·c)]; only ⊗ flattens, not the dot.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* dot = make_dot(ctx, b, c);
    auto const* e = make_tensor_product(ctx, a, dot);
    auto pp = flat(+1, e);
    EXPECT_EQ(pp.factors, (std::vector<Expr const*>{a, dot}));
}

// ---- factor encapsulation ----------------------------------------------

TEST(Encapsulate, BareTensorObjectBecomesAtom)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto sf = encapsulate(ctx, a);
    EXPECT_EQ(sf.sign, +1);
    ASSERT_TRUE(std::holds_alternative<Atom>(sf.factor->node));
    EXPECT_EQ(std::get<Atom>(sf.factor->node).obj.name.v.view(), "a");
}

TEST(Encapsulate, DotChainBecomesFlatContraction)
{
    // (A·B)·c and A·(B·c) both flatten to factors [A,B,c], ops [Dot,Dot].
    Context ctx;
    auto const* A = atomr(ctx, "A", 2);
    auto const* B = atomr(ctx, "B", 2);
    auto const* c = atom(ctx, "c");
    auto const* left = make_dot(ctx, make_dot(ctx, A, B), c);
    auto const* right = make_dot(ctx, A, make_dot(ctx, B, c));
    for (auto const* e: {left, right})
    {
        auto sf = encapsulate(ctx, e);
        EXPECT_EQ(sf.sign, +1);
        ASSERT_TRUE(std::holds_alternative<Contraction>(sf.factor->node));
        auto const& con = std::get<Contraction>(sf.factor->node);
        ASSERT_EQ(con.factors.size(), 3u);
        EXPECT_EQ(con.ops, (std::vector<COp>{COp::Dot, COp::Dot}));
        EXPECT_EQ(std::get<Atom>(con.factors[0]->node).obj.name.v.view(), "A");
        EXPECT_EQ(std::get<Atom>(con.factors[2]->node).obj.name.v.view(), "c");
    }
}

TEST(Encapsulate, DDotOperatorRecorded)
{
    Context ctx;
    auto const* A = atomr(ctx, "A", 2);
    auto const* B = atomr(ctx, "B", 2);
    auto sf = encapsulate(ctx, make_ddot(ctx, A, B));
    auto const& con = std::get<Contraction>(sf.factor->node);
    EXPECT_EQ(con.ops, (std::vector<COp>{COp::DDot}));
}

TEST(Encapsulate, GenuineSumBecomesParen)
{
    // A genuine sum factor encapsulates into a Paren over the recursively
    // canonicalized interior (b + c → Nf with two terms).
    Context ctx;
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto sf = encapsulate(ctx, make_sum(ctx, b, c));
    EXPECT_EQ(sf.sign, +1);
    ASSERT_TRUE(std::holds_alternative<Paren>(sf.factor->node));
    EXPECT_EQ(std::get<Paren>(sf.factor->node).body->terms.size(), 2u);
}

TEST(Encapsulate, UnsupportedNodeThrows)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    // A bare ⊗ inside an operand awaits fence distribution — encapsulate alone
    // does not split it.
    EXPECT_THROW(
        (void)encapsulate(ctx, make_tensor_product(ctx, a, b)),
        std::invalid_argument);
}

// ---- unary invariants --------------------------------------------------

TEST(EncapsulateUnary, TraceTransposeVec)
{
    Context ctx;
    auto const* A = atomr(ctx, "A", 2);
    struct Case
    {
        Expr const* (*make)(Context&, Expr const*);
        UnaryOp op;
    };
    for (auto const& cse:
         {Case{&make_trace, UnaryOp::Trace},
          Case{&make_vector_invariant, UnaryOp::VectorInvariant},
          Case{&make_transpose, UnaryOp::Transpose}})
    {
        auto sf = encapsulate(ctx, cse.make(ctx, A));
        EXPECT_EQ(sf.sign, +1);
        ASSERT_TRUE(std::holds_alternative<Unary>(sf.factor->node));
        auto const& u = std::get<Unary>(sf.factor->node);
        EXPECT_EQ(u.op, cse.op);
        EXPECT_EQ(std::get<Atom>(u.operand->node).obj.name.v.view(), "A");
    }
}

TEST(EncapsulateUnary, TraceIsScalarVecAndTransposeAreTensors)
{
    // tr(A) rank 0 → scalars; vec(A) rank 1 and A^T rank 2 → tensors.
    Context ctx;
    auto const* A = atomr(ctx, "A", 2);
    auto tr =
        place_factors(ctx, ProductParts{Rational{1}, {make_trace(ctx, A)}});
    EXPECT_EQ(tr.scalars.size(), 1u);
    EXPECT_TRUE(tr.tensors.empty());

    auto vt = place_factors(
        ctx,
        ProductParts{
            Rational{1},
            {make_vector_invariant(ctx, A), make_transpose(ctx, A)}});
    EXPECT_TRUE(vt.scalars.empty());
    EXPECT_EQ(vt.tensors.size(), 2u);
}

// ---- cross encapsulation (C6) ------------------------------------------

TEST(EncapsulateCross, BinaryInCanonicalOrderKeepsSign)
{
    // a×b with a<b: stays a×b, sign +1.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto sf = encapsulate(ctx, make_cross(ctx, a, b));
    EXPECT_EQ(sf.sign, +1);
    ASSERT_TRUE(std::holds_alternative<nf::Cross>(sf.factor->node));
    auto const& cr = std::get<nf::Cross>(sf.factor->node);
    ASSERT_EQ(cr.factors.size(), 2u);
    EXPECT_EQ(std::get<Atom>(cr.factors[0]->node).obj.name.v.view(), "a");
    EXPECT_EQ(std::get<Atom>(cr.factors[1]->node).obj.name.v.view(), "b");
}

TEST(EncapsulateCross, BinaryOutOfOrderFlipsSignAndReorders)
{
    // b×a with a<b: becomes a×b, sign -1  (a×b = -(b×a)).
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto sf = encapsulate(ctx, make_cross(ctx, b, a));
    EXPECT_EQ(sf.sign, -1);
    auto const& cr = std::get<nf::Cross>(sf.factor->node);
    EXPECT_EQ(std::get<Atom>(cr.factors[0]->node).obj.name.v.view(), "a");
    EXPECT_EQ(std::get<Atom>(cr.factors[1]->node).obj.name.v.view(), "b");
}

TEST(EncapsulateCross, AnticommutationSignLandsInCoeff)
{
    // place_factors lifts the -1 from b×a into the term coeff.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto t =
        place_factors(ctx, ProductParts{Rational{1}, {make_cross(ctx, b, a)}});
    EXPECT_EQ(t.coeff, Rational{-1});
    ASSERT_EQ(t.tensors.size(), 1u); // a×b is rank 1
    EXPECT_TRUE(std::holds_alternative<nf::Cross>(t.tensors[0]->node));
}

TEST(EncapsulateCross, RankTwoFenceReassociates)
{
    // (a × M) × b  with M rank 2  →  a × (M × b)  (000055 fence).
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* M = atomr(ctx, "M", 2);
    auto const* e = make_cross(ctx, make_cross(ctx, a, M), b);
    auto sf = encapsulate(ctx, e);
    // Top level is a × (M×b): factors [a, (M×b)].
    auto const& top = std::get<nf::Cross>(sf.factor->node);
    ASSERT_EQ(top.factors.size(), 2u);
    EXPECT_EQ(std::get<Atom>(top.factors[0]->node).obj.name.v.view(), "a");
    ASSERT_TRUE(std::holds_alternative<nf::Cross>(top.factors[1]->node));
    auto const& inner = std::get<nf::Cross>(top.factors[1]->node);
    EXPECT_EQ(std::get<Atom>(inner.factors[0]->node).obj.name.v.view(), "M");
    EXPECT_EQ(std::get<Atom>(inner.factors[1]->node).obj.name.v.view(), "b");
}

// ---- interior commutative ordering (C7) --------------------------------

TEST(EncapsulateInterior, RankOneDotIsOrdered)
{
    // b·a (rank-1 vectors) → operands ordered a, b (no sign change).
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto sf = encapsulate(ctx, make_dot(ctx, b, a));
    EXPECT_EQ(sf.sign, +1);
    auto const& con = std::get<Contraction>(sf.factor->node);
    EXPECT_EQ(std::get<Atom>(con.factors[0]->node).obj.name.v.view(), "a");
    EXPECT_EQ(std::get<Atom>(con.factors[1]->node).obj.name.v.view(), "b");
}

TEST(EncapsulateInterior, MatrixVectorDotDoesNotReorder)
{
    // A·b (rank-2 · rank-1) is not commutative: order is preserved.
    Context ctx;
    auto const* A = atomr(ctx, "A", 2);
    auto const* b = atom(ctx, "b");
    auto sf = encapsulate(ctx, make_dot(ctx, A, b));
    auto const& con = std::get<Contraction>(sf.factor->node);
    EXPECT_EQ(std::get<Atom>(con.factors[0]->node).obj.name.v.view(), "A");
    EXPECT_EQ(std::get<Atom>(con.factors[1]->node).obj.name.v.view(), "b");
}

TEST(EncapsulateInterior, DoubleDotOfRankTwoIsOrdered)
{
    // B:A with both rank 2 → ordered A:B (scalar, symmetric).
    Context ctx;
    auto const* A = atomr(ctx, "A", 2);
    auto const* B = atomr(ctx, "B", 2);
    auto sf = encapsulate(ctx, make_ddot(ctx, B, A));
    auto const& con = std::get<Contraction>(sf.factor->node);
    EXPECT_EQ(con.ops, (std::vector<COp>{COp::DDot}));
    EXPECT_EQ(std::get<Atom>(con.factors[0]->node).obj.name.v.view(), "A");
    EXPECT_EQ(std::get<Atom>(con.factors[1]->node).obj.name.v.view(), "B");
}

TEST(EncapsulateInterior, HigherRankDoubleDotDoesNotReorder)
{
    // C:ε with C rank 4, ε rank 2 (stress = stiffness : strain) is directional:
    // C:ε ≠ ε:C, so the order is preserved even though "C" sorts after "A".
    Context ctx;
    auto const* C = atomr(ctx, "C", 4);
    auto const* eps = atomr(ctx, "A", 2); // sorts before "C"
    auto sf = encapsulate(ctx, make_ddot(ctx, C, eps));
    auto const& con = std::get<Contraction>(sf.factor->node);
    EXPECT_EQ(std::get<Atom>(con.factors[0]->node).obj.name.v.view(), "C");
    EXPECT_EQ(std::get<Atom>(con.factors[1]->node).obj.name.v.view(), "A");
}

// ---- region placement --------------------------------------------------

TEST(PlaceFactors, RankOneGoesToTensors)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto t = place_factors(ctx, ProductParts{Rational{2}, {a}});
    EXPECT_EQ(t.coeff, Rational{2});
    EXPECT_TRUE(t.scalars.empty());
    ASSERT_EQ(t.tensors.size(), 1u);
    EXPECT_TRUE(std::holds_alternative<Atom>(t.tensors[0]->node));
}

TEST(PlaceFactors, ScalarContractionGoesToScalars)
{
    // a·b is rank 0 ⇒ scalars region.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto t =
        place_factors(ctx, ProductParts{Rational{1}, {make_dot(ctx, a, b)}});
    ASSERT_EQ(t.scalars.size(), 1u);
    EXPECT_TRUE(t.tensors.empty());
    EXPECT_TRUE(std::holds_alternative<Contraction>(t.scalars[0]->node));
}

TEST(PlaceFactors, WedgedScalarFloatsOutOfTensorRun)
{
    // [a, (b·c), d] with a,d rank-1 and (b·c) rank-0:
    // the scalar lands in `scalars`, the two legs in `tensors` (adjacent) —
    // the 000056 wedged-scalar fold failure dissolves here.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* d = atom(ctx, "d");
    auto t = place_factors(
        ctx, ProductParts{Rational{1}, {a, make_dot(ctx, b, c), d}});
    ASSERT_EQ(t.scalars.size(), 1u);
    EXPECT_TRUE(std::holds_alternative<Contraction>(t.scalars[0]->node));
    ASSERT_EQ(t.tensors.size(), 2u);
    EXPECT_EQ(std::get<Atom>(t.tensors[0]->node).obj.name.v.view(), "a");
    EXPECT_EQ(std::get<Atom>(t.tensors[1]->node).obj.name.v.view(), "d");
}

TEST(PlaceFactors, UnknownRankGoesToTensors)
{
    Context ctx;
    // A tensor object with no declared rank ⇒ infer_rank is nullopt; it lands
    // positionally in the tensor region (not the commutative scalar region).
    auto const* x = make_tensor_object(ctx, make_tensor_name("X"));
    auto t = place_factors(ctx, ProductParts{Rational{1}, {x}});
    EXPECT_TRUE(t.scalars.empty());
    ASSERT_EQ(t.tensors.size(), 1u);
    EXPECT_EQ(std::get<Atom>(t.tensors[0]->node).obj.name.v.view(), "X");
}

TEST(PlaceFactors, ScalarsAreSortedTensorsArePositional)
{
    // Scalars q, p (rank 0) come out sorted [p, q]; the rank-1 legs keep order.
    Context ctx;
    auto const* p = atomr(ctx, "p", 0);
    auto const* q = atomr(ctx, "q", 0);
    auto const* x = atom(ctx, "x");
    auto const* y = atom(ctx, "y");
    auto t = place_factors(ctx, ProductParts{Rational{1}, {q, x, p, y}});
    ASSERT_EQ(t.scalars.size(), 2u);
    EXPECT_EQ(std::get<Atom>(t.scalars[0]->node).obj.name.v.view(), "p");
    EXPECT_EQ(std::get<Atom>(t.scalars[1]->node).obj.name.v.view(), "q");
    ASSERT_EQ(t.tensors.size(), 2u);
    EXPECT_EQ(std::get<Atom>(t.tensors[0]->node).obj.name.v.view(), "x");
    EXPECT_EQ(std::get<Atom>(t.tensors[1]->node).obj.name.v.view(), "y");
}

// ---- per-term lowering (⊗-fence distribution) --------------------------

namespace
{
auto term1(Context& ctx, Expr const* body) -> Term
{
    return lower_term(ctx, SignedExpr{+1, body});
}
} // namespace

TEST(LowerTerm, PlainProduct)
{
    // 2·a⊗b  ->  coeff 2, tensors [a, b].
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* e = make_tensor_product(
        ctx, make_scalar(ctx, Rational{2}), make_tensor_product(ctx, a, b));
    auto t = term1(ctx, e);
    EXPECT_EQ(t.coeff, Rational{2});
    ASSERT_EQ(t.tensors.size(), 2u);
    EXPECT_EQ(std::get<Atom>(t.tensors[0]->node).obj.name.v.view(), "a");
    EXPECT_EQ(std::get<Atom>(t.tensors[1]->node).obj.name.v.view(), "b");
}

TEST(LowerTerm, DotFenceDistributesTensorProductOut)
{
    // A·(b⊗c)  ->  (A·b)⊗c  ->  tensors [(A·b), c], no ⊗ buried in a
    // contraction.
    Context ctx;
    auto const* A = atomr(ctx, "A", 2);
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto t = term1(ctx, make_dot(ctx, A, make_tensor_product(ctx, b, c)));
    EXPECT_TRUE(t.scalars.empty());
    ASSERT_EQ(t.tensors.size(), 2u);
    ASSERT_TRUE(std::holds_alternative<Contraction>(t.tensors[0]->node));
    auto const& con = std::get<Contraction>(t.tensors[0]->node);
    ASSERT_EQ(con.factors.size(), 2u);
    EXPECT_EQ(std::get<Atom>(con.factors[0]->node).obj.name.v.view(), "A");
    EXPECT_EQ(std::get<Atom>(con.factors[1]->node).obj.name.v.view(), "b");
    EXPECT_EQ(std::get<Atom>(t.tensors[1]->node).obj.name.v.view(), "c");
}

TEST(LowerTerm, CrossFenceDistributesTensorProductOut)
{
    // a×(b⊗c)  ->  (a×b)⊗c  ->  tensors [(a×b), c].
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto t = term1(ctx, make_cross(ctx, a, make_tensor_product(ctx, b, c)));
    ASSERT_EQ(t.tensors.size(), 2u);
    EXPECT_TRUE(std::holds_alternative<nf::Cross>(t.tensors[0]->node));
    EXPECT_EQ(std::get<Atom>(t.tensors[1]->node).obj.name.v.view(), "c");
}

TEST(LowerTerm, GenuineSumOperandBecomesParen)
{
    // A·(b+c): the sum is NOT distributed (explicit transform only); the sum
    // operand becomes a Paren, so the term is the contraction A·(b+c) with a
    // Paren second operand.
    Context ctx;
    auto const* A = atomr(ctx, "A", 2);
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto t = term1(ctx, make_dot(ctx, A, make_sum(ctx, b, c)));
    ASSERT_EQ(t.tensors.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<Contraction>(t.tensors[0]->node));
    auto const& con = std::get<Contraction>(t.tensors[0]->node);
    ASSERT_EQ(con.factors.size(), 2u);
    EXPECT_TRUE(std::holds_alternative<Atom>(con.factors[0]->node));
    EXPECT_TRUE(std::holds_alternative<Paren>(con.factors[1]->node));
    EXPECT_EQ(std::get<Paren>(con.factors[1]->node).body->terms.size(), 2u);
}

// ---- summation resolution (C8) -----------------------------------------

namespace
{

// An indexed rank-1 vector `name` carrying `idx` at level `lvl` (Oblique, 3D).
auto ivec(Context& ctx, std::string_view name, Level lvl, CountableIndex idx)
    -> Expr const*
{
    return make_tensor_object(
        ctx,
        make_tensor_name(name),
        {SlotBinding{
            IndexSlot{lvl, Realm::Oblique, space_3d()}, IndexAssoc{idx}}},
        1);
}

// The CountableIndex id in the first slot of an Atom factor.
auto slot0_id(Factor const* f) -> int
{
    auto const& obj = std::get<Atom>(f->node).obj;
    return std::get<CountableIndex>(*obj.slots.at(0).index).id;
}

} // namespace

TEST(Summation, ImplicitObliqueContraction)
{
    // a^i b_i: one realm-implicit dummy, mode Default, α-renamed to -1; both
    // factors carry the renamed id.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* a = ivec(ctx, "a", Level::Upper, i);
    auto const* b = ivec(ctx, "b", Level::Lower, i);
    auto t = term1(ctx, make_tensor_product(ctx, a, b));
    ASSERT_EQ(t.bound.size(), 1u);
    EXPECT_EQ(t.bound[0].index.id, -1);
    EXPECT_EQ(t.bound[0].mode, SumMode::Default);
    ASSERT_EQ(t.tensors.size(), 2u);
    EXPECT_EQ(slot0_id(t.tensors[0]), -1);
    EXPECT_EQ(slot0_id(t.tensors[1]), -1);
}

TEST(Summation, ExplicitSumOnRealmDefaultNormalizesToDefault)
{
    // Σ_i (a^i b_i): the explicit Σ merely confirms the oblique default, so it
    // normalizes to mode Default — identical to the implicit form above.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* a = ivec(ctx, "a", Level::Upper, i);
    auto const* b = ivec(ctx, "b", Level::Lower, i);
    auto t =
        term1(ctx, make_explicit_sum(ctx, i, make_tensor_product(ctx, a, b)));
    ASSERT_EQ(t.bound.size(), 1u);
    EXPECT_EQ(t.bound[0].index.id, -1);
    EXPECT_EQ(t.bound[0].mode, SumMode::Default);
}

TEST(Summation, ExplicitSumNotRealmDefaultIsSum)
{
    // Σ_i a_i: a single occurrence summed — the realm rule would NOT contract,
    // so the explicit Σ is preserved as mode Sum (still α-renamed to -1).
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* a = ivec(ctx, "a", Level::Lower, i);
    auto t = term1(ctx, make_explicit_sum(ctx, i, a));
    ASSERT_EQ(t.bound.size(), 1u);
    EXPECT_EQ(t.bound[0].index.id, -1);
    EXPECT_EQ(t.bound[0].mode, SumMode::Sum);
    ASSERT_EQ(t.tensors.size(), 1u);
    EXPECT_EQ(slot0_id(t.tensors[0]), -1);
}

TEST(Summation, NoSumSuppressesContractionAndKeepsFreeId)
{
    // NoSum_i (a^i b_i): NoSum suppresses the oblique default; i stays free
    // (its original id, not α-renamed) and is recorded with mode NoSum.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* a = ivec(ctx, "a", Level::Upper, i);
    auto const* b = ivec(ctx, "b", Level::Lower, i);
    auto t = term1(ctx, make_no_sum(ctx, i, make_tensor_product(ctx, a, b)));
    ASSERT_EQ(t.bound.size(), 1u);
    EXPECT_EQ(t.bound[0].index.id, i.id); // original free id, not -1
    EXPECT_EQ(t.bound[0].mode, SumMode::NoSum);
    ASSERT_EQ(t.tensors.size(), 2u);
    EXPECT_EQ(slot0_id(t.tensors[0]), i.id);
    EXPECT_EQ(slot0_id(t.tensors[1]), i.id);
}

TEST(Summation, TwoDummiesAreFubiniInvariant)
{
    // a^i b_i c^j d_j and its i↔j relabelling canonicalize equal: the two
    // dummies are interchangeable (Fubini), both Default, renamed to -1 / -2.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto prod = [&](CountableIndex x, CountableIndex y)
    {
        return make_tensor_product(
            ctx,
            make_tensor_product(
                ctx,
                ivec(ctx, "a", Level::Upper, x),
                ivec(ctx, "b", Level::Lower, x)),
            make_tensor_product(
                ctx,
                ivec(ctx, "c", Level::Upper, y),
                ivec(ctx, "d", Level::Lower, y)));
    };
    auto t1 = term1(ctx, prod(i, j));
    auto t2 = term1(ctx, prod(j, i));
    ASSERT_EQ(t1.bound.size(), 2u);
    EXPECT_EQ(t1.bound[0].index.id, -1);
    EXPECT_EQ(t1.bound[1].index.id, -2);
    EXPECT_EQ(t1.bound[0].mode, SumMode::Default);
    EXPECT_EQ(t1.bound[1].mode, SumMode::Default);
    EXPECT_TRUE(equal(t1, t2));
}

TEST(Summation, RangedExplicitSumCarriesRange)
{
    // Σ_{i∈n} a_i: a symbolic-bound sum lowers to a Sum-mode dummy (α-renamed
    // to -1) carrying the range; the canonical form round-trips.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* a = ivec(ctx, "a", Level::Lower, i);
    auto const* n = make_tensor_object(ctx, make_tensor_name("n"));
    auto t = term1(ctx, make_explicit_sum(ctx, i, a, n));
    ASSERT_EQ(t.bound.size(), 1u);
    EXPECT_EQ(t.bound[0].index.id, -1);
    EXPECT_EQ(t.bound[0].mode, SumMode::Sum);
    ASSERT_NE(t.bound[0].range, nullptr);
    // The range is the canonical Nf of the symbol n (one tensor term).
    ASSERT_EQ(t.bound[0].range->terms.size(), 1u);
}

// ---- like-term collection (C9) -----------------------------------------

namespace
{

// A single-tensor term `coeff · name` (name an abstract rank-1 tensor).
auto scaled(Context& ctx, Rational coeff, std::string_view name) -> Term
{
    Term t;
    t.coeff = coeff;
    t.tensors.push_back(
        make_atom(ctx, std::get<TensorObject>(atom(ctx, name)->node)));
    return t;
}

auto tname(Term const& t, std::size_t i) -> std::string_view
{
    return std::get<Atom>(t.tensors.at(i)->node).obj.name.v.view();
}

} // namespace

TEST(CollectTerms, MergesLikeTerms)
{
    // 2a + 3a → 5a  (distinct factor instances, same key → one term).
    Context ctx;
    auto out = collect_terms(
        {scaled(ctx, Rational{2}, "a"), scaled(ctx, Rational{3}, "a")});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].coeff, Rational{5});
    ASSERT_EQ(out[0].tensors.size(), 1u);
    EXPECT_EQ(tname(out[0], 0), "a");
}

TEST(CollectTerms, CancelsOpposite)
{
    // a + (−a) → 0  (empty term set).
    Context ctx;
    auto out = collect_terms(
        {scaled(ctx, Rational{1}, "a"), scaled(ctx, Rational{-1}, "a")});
    EXPECT_TRUE(out.empty());
}

TEST(CollectTerms, MergeThenCancel)
{
    // a + a − 2a → 0.
    Context ctx;
    auto out = collect_terms(
        {scaled(ctx, Rational{1}, "a"),
         scaled(ctx, Rational{1}, "a"),
         scaled(ctx, Rational{-2}, "a")});
    EXPECT_TRUE(out.empty());
}

TEST(CollectTerms, MergesAndSortsTogether)
{
    // b + a + 2a → [3a, b]  (merge like terms, drop nothing, canonical order).
    Context ctx;
    auto out = collect_terms(
        {scaled(ctx, Rational{1}, "b"),
         scaled(ctx, Rational{1}, "a"),
         scaled(ctx, Rational{2}, "a")});
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(tname(out[0], 0), "a");
    EXPECT_EQ(out[0].coeff, Rational{3});
    EXPECT_EQ(tname(out[1], 0), "b");
    EXPECT_EQ(out[1].coeff, Rational{1});
}

TEST(CollectTerms, EmptyIsZero)
{
    EXPECT_TRUE(collect_terms({}).empty());
}

// ---- canonicalize_nf entry point (C10) ---------------------------------

TEST(CanonicalizeNf, SingleAtom)
{
    Context ctx;
    auto const* nf = canonicalize_nf(ctx, atom(ctx, "a"));
    ASSERT_EQ(nf->terms.size(), 1u);
    EXPECT_EQ(nf->terms[0].coeff, Rational{1});
    ASSERT_EQ(nf->terms[0].tensors.size(), 1u);
    EXPECT_EQ(tname(nf->terms[0], 0), "a");
}

TEST(CanonicalizeNf, SumCancelsAcrossTerms)
{
    // (a + b) − a → b  (sign-drift cancellation is structural).
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* nf =
        canonicalize_nf(ctx, make_difference(ctx, make_sum(ctx, a, b), a));
    ASSERT_EQ(nf->terms.size(), 1u);
    EXPECT_EQ(tname(nf->terms[0], 0), "b");
    EXPECT_EQ(nf->terms[0].coeff, Rational{1});
}

TEST(CanonicalizeNf, LikeTermsMerge)
{
    // 2a + 3a → 5a
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto sa = [&](int n)
    { return make_tensor_product(ctx, make_scalar(ctx, Rational{n}), a); };
    auto const* nf = canonicalize_nf(ctx, make_sum(ctx, sa(2), sa(3)));
    ASSERT_EQ(nf->terms.size(), 1u);
    EXPECT_EQ(nf->terms[0].coeff, Rational{5});
}

TEST(CanonicalizeNf, WedgedScalarFloatsOut)
{
    // (a·b) ⊗ C: the scalar a·b floats to the scalar region; C is the tensor.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* C = atomr(ctx, "C", 2);
    auto const* nf =
        canonicalize_nf(ctx, make_tensor_product(ctx, make_dot(ctx, a, b), C));
    ASSERT_EQ(nf->terms.size(), 1u);
    auto const& t = nf->terms[0];
    EXPECT_EQ(t.scalars.size(), 1u); // a·b
    ASSERT_EQ(t.tensors.size(), 1u); // C
    EXPECT_EQ(tname(t, 0), "C");
}

TEST(CanonicalizeNf, EmptyIsZero)
{
    // a − a → 0 (empty term set).
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* nf = canonicalize_nf(ctx, make_difference(ctx, a, a));
    EXPECT_TRUE(nf->terms.empty());
}

TEST(CanonicalizeNf, ParenSumRecurses)
{
    // A·(b+c): the genuine sum sinks into a Paren whose interior canonicalizes.
    Context ctx;
    auto const* A = atomr(ctx, "A", 2);
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* nf =
        canonicalize_nf(ctx, make_dot(ctx, A, make_sum(ctx, b, c)));
    ASSERT_EQ(nf->terms.size(), 1u);
    auto const& con = std::get<Contraction>(nf->terms[0].tensors[0]->node);
    EXPECT_TRUE(std::holds_alternative<Paren>(con.factors[1]->node));
    EXPECT_EQ(std::get<Paren>(con.factors[1]->node).body->terms.size(), 2u);
}

// The new lowering maps an expr and its old-canonical form to the same `Nf`:
// `canonicalize_nf(e) == canonicalize_nf(canonicalize(e))`.  Old canon is
// semantics-preserving, so any divergence is a real disagreement (a bug or a
// signed-off improvement) — the C10 differential check.
TEST(CanonicalizeNf, DifferentialVsOldCanon)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* C = atomr(ctx, "C", 2);
    CountableIndex i{ctx.alloc_index_id()};
    auto sa = [&](int n)
    { return make_tensor_product(ctx, make_scalar(ctx, Rational{n}), a); };

    std::vector<Expr const*> corpus = {
        a,
        make_dot(ctx, a, b),
        make_dot(ctx, b, a), // commutes → same form
        make_tensor_product(ctx, make_dot(ctx, a, b), C), // wedged scalar
        make_difference(ctx, make_sum(ctx, a, b), a),     // a+b−a → b
        make_sum(ctx, sa(2), sa(3)),                      // 2a+3a → 5a
        make_tensor_product( // a^i b_i (implicit Σ)
            ctx,
            ivec(ctx, "a", Level::Upper, i),
            ivec(ctx, "b", Level::Lower, i)),
    };
    for (std::size_t n = 0; n < corpus.size(); ++n)
    {
        auto const* e = corpus[n];
        auto const* via_raw = canonicalize_nf(ctx, e);
        auto const* via_canon =
            canonicalize_nf(ctx, steps::canonicalize(ctx, e));
        EXPECT_TRUE(equal(*via_raw, *via_canon)) << "corpus #" << n;
    }
}

// ---- raise round-trip (C12) --------------------------------------------

namespace
{
// canonicalize_nf ∘ raise == id on canonical Nfs (the idempotence property).
auto roundtrips(Context& ctx, Expr const* e) -> bool
{
    auto const* nf1 = canonicalize_nf(ctx, e);
    auto const* nf2 = canonicalize_nf(ctx, raise(ctx, *nf1));
    return equal(*nf1, *nf2);
}
} // namespace

TEST(Raise, EmptyNfIsZero)
{
    // a − a → empty Nf → raises to the literal 0.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* nf = canonicalize_nf(ctx, make_difference(ctx, a, a));
    ASSERT_TRUE(nf->terms.empty());
    auto const* e = raise(ctx, *nf);
    ASSERT_TRUE(std::holds_alternative<ScalarLiteral>(e->node));
    EXPECT_EQ(std::get<ScalarLiteral>(e->node).value, Rational{0});
}

TEST(Raise, RoundTripCorpus)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* A = atomr(ctx, "A", 2);
    auto const* B = atomr(ctx, "B", 2);
    auto const* C = atomr(ctx, "C", 2);
    CountableIndex i{ctx.alloc_index_id()};
    auto sc = [&](int n, Expr const* x)
    { return make_tensor_product(ctx, make_scalar(ctx, Rational{n}), x); };

    std::vector<Expr const*> corpus = {
        a,                                                // bare atom
        sc(2, a),                                         // 2a
        make_negate(ctx, a),                              // −a
        sc(-2, a),                                        // −2a
        make_scalar(ctx, Rational{5}),                    // pure scalar term
        make_difference(ctx, make_sum(ctx, a, b), a),     // a+b−a → b
        make_dot(ctx, a, b),                              // scalar contraction
        make_tensor_product(ctx, make_dot(ctx, a, b), C), // wedged scalar
        make_cross(ctx, a, b),                            // cross
        make_tensor_product(ctx, make_cross(ctx, a, b), C), // juxtaposed cross
        make_dot(ctx, A, make_sum(ctx, b, c)),              // Paren
        make_trace(ctx, A),                                 // unary scalar
        make_transpose(ctx, A),                             // unary tensor
        make_vector_invariant(ctx, A),                      // vec
        make_dot(ctx, make_dot(ctx, A, B), a),              // 3-factor chain
        make_sum(ctx, a, make_difference(ctx, sc(2, b), make_trace(ctx, A))),
        make_tensor_product( // a^i b_i (implicit Default)
            ctx,
            ivec(ctx, "a", Level::Upper, i),
            ivec(ctx, "b", Level::Lower, i)),
        make_explicit_sum( // Σ_i a_i (Sum mode)
            ctx,
            i,
            ivec(ctx, "a", Level::Lower, i)),
        make_no_sum( // NoSum_i a^i b_i (NoSum mode)
            ctx,
            i,
            make_tensor_product(
                ctx,
                ivec(ctx, "a", Level::Upper, i),
                ivec(ctx, "b", Level::Lower, i))),
    };
    for (std::size_t n = 0; n < corpus.size(); ++n)
        EXPECT_TRUE(roundtrips(ctx, corpus[n])) << "corpus #" << n;
}

// ---- abstract (rank-less) tensors (C13a) -------------------------------

TEST(CanonicalizeNf, RanklessTensorsSumCommutes)
{
    // A + B with abstract, rank-less tensors: no throw, and the additive layer
    // orders the terms so the two orderings agree.
    Context ctx;
    auto const* A = make_tensor_object(ctx, make_tensor_name("A"));
    auto const* B = make_tensor_object(ctx, make_tensor_name("B"));
    EXPECT_TRUE(equal(
        *canonicalize_nf(ctx, make_sum(ctx, A, B)),
        *canonicalize_nf(ctx, make_sum(ctx, B, A))));
}

// ---- symmetry-orbit canonicalization in the Nf path (C13b) -------------

namespace
{
auto eps3(Context& ctx, Level lvl, IndexAssoc a, IndexAssoc b, IndexAssoc c)
    -> Expr const*
{
    return make_levi_civita(
        ctx, Realm::Oblique, space_3d(), {lvl, lvl, lvl}, {a, b, c});
}
} // namespace

TEST(NfSymmetry, DeltaSlotSwapCanonicalizesEqual)
{
    // δ is symmetric: δ^a_b and δ_b^a canonicalize to the same orbit-minimal
    // Nf.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    auto const* d_ab =
        make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, a, b);
    auto const* d_ba =
        make_delta(ctx, Realm::Oblique, sp, Level::Lower, Level::Upper, b, a);
    EXPECT_TRUE(
        equal(*canonicalize_nf(ctx, d_ab), *canonicalize_nf(ctx, d_ba)));
}

TEST(NfSymmetry, EpsTranspositionCancels)
{
    // Odd permutation flips sign: ε^{ijk} + ε^{jik} → 0 (empty Nf).
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    auto const* ijk =
        eps3(ctx, Level::Upper, IndexAssoc{i}, IndexAssoc{j}, IndexAssoc{k});
    auto const* jik =
        eps3(ctx, Level::Upper, IndexAssoc{j}, IndexAssoc{i}, IndexAssoc{k});
    auto const* nf = canonicalize_nf(ctx, make_sum(ctx, ijk, jik));
    EXPECT_TRUE(nf->terms.empty());
}

TEST(NfSymmetry, EpsRepeatedConcreteIndexIsZero)
{
    // An arrangement reachable with both signs is identically zero: ε^{11k} →
    // 0.
    Context ctx;
    auto const* z = eps3(
        ctx,
        Level::Upper,
        IndexAssoc{ConcreteIndex{1}},
        IndexAssoc{ConcreteIndex{1}},
        IndexAssoc{ConcreteIndex{2}});
    auto const* nf = canonicalize_nf(ctx, z);
    EXPECT_TRUE(nf->terms.empty());
}

// ---- double-dot fence distribution (C13c) ------------------------------

TEST(CanonicalizeNf, DoubleDotOfDyadsExpands)
{
    // (a⊗b):(c⊗d) → (a·c)(b·d): two scalar dots, no ⊗ buried in a `:` operand.
    Context ctx;
    auto const* e = make_ddot(
        ctx,
        make_tensor_product(ctx, atom(ctx, "a"), atom(ctx, "b")),
        make_tensor_product(ctx, atom(ctx, "c"), atom(ctx, "d")));
    auto const* nf = canonicalize_nf(ctx, e);
    ASSERT_EQ(nf->terms.size(), 1u);
    EXPECT_EQ(nf->terms[0].scalars.size(), 2u); // (a·c) and (b·d)
    EXPECT_TRUE(nf->terms[0].tensors.empty());
}

TEST(CanonicalizeNf, ZeroFenceInsideContractionFolds)
{
    // a·(b⊗0) → 0: a literal-0 operand buried inside a ⊗ fence inside a
    // contraction (vibe 000078).  Differentiating a constant frame vector
    // (∂_i e_j = 0) leaves exactly this shape — `e_j ⊗ (0·∂_iε)` — which
    // `encapsulate` has no ⊗ arm for; the algebraic zero law must collapse the
    // whole term first.  Before the fold this *threw*; now it vanishes.
    Context ctx;
    auto const* fence =
        make_tensor_product(ctx, atom(ctx, "b"), make_scalar(ctx, Rational{0}));
    auto const* e = make_dot(ctx, atom(ctx, "a"), fence);
    auto const* nf = canonicalize_nf(ctx, e);
    EXPECT_TRUE(nf->terms.empty()); // 0 has no terms
}

TEST(CanonicalizeNf, RightNestedFanInKeepsLegTopology)
{
    // a·(b·T) is a *scalar* (a_j b_i T_ij): the vector b is fully consumed
    // contracting into T, so a fans onto T's other leg — a right-nesting the
    // flat left-fold chain must NOT reassociate to (a·b)·T (vibe 000078 bug 3b,
    // which mis-flattened it to a rank-2 `a·b T`).  The faithful chain is
    // b·T·a, so canon must agree with the independently built b·(T·a).
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* T = atomr(ctx, "T", 2);
    auto const* nested = make_dot(ctx, a, make_dot(ctx, b, T));
    auto const* canon = steps::canonicalize(ctx, nested);
    EXPECT_EQ(infer_rank(canon), std::optional<int>{0}); // scalar, not rank 2
    // b·(T·a) is the same tensor built without the fan-in nesting.
    auto const* oracle =
        steps::canonicalize(ctx, make_dot(ctx, b, make_dot(ctx, T, a)));
    EXPECT_TRUE(structural_eq(canon, oracle));
    // …and genuinely T-oriented: b·(Tᵀ·a) (the transpose) must differ.
    auto const* wrong = steps::canonicalize(
        ctx, make_dot(ctx, b, make_dot(ctx, make_transpose(ctx, T), a)));
    EXPECT_FALSE(structural_eq(canon, wrong));
}

TEST(CanonicalizeNf, SymmetricTransposeFolds)
{
    // εᵀ = ε for a symmetric rank-2 tensor (the slot swap is a symmetry); a
    // general rank-2 keeps an explicit transpose (vibe 000078).  The invariant
    // form carries no slots, so the symmetry is probed via synthetic ones.
    Context ctx;
    auto const* eps =
        make_field(ctx, make_tensor_name("\\varepsilon"), 2, {}, /*sym=*/true);
    auto const* gen = make_field(ctx, make_tensor_name("G"), 2, {});
    EXPECT_TRUE(structural_eq(
        steps::canonicalize(ctx, make_transpose(ctx, eps)),
        steps::canonicalize(ctx, eps)));
    EXPECT_FALSE(structural_eq(
        steps::canonicalize(ctx, make_transpose(ctx, gen)),
        steps::canonicalize(ctx, gen)));
}

TEST(CanonicalizeNf, RankTwoFanInInsertsTranspose)
{
    // T·(a·S), T,S rank 2: a·S is a vector on S's free leg, so T fans onto S's
    // *second* leg — faithfully T·Sᵀ·a (a transpose the flattener must emit).
    // Cross-checked against T·(Sᵀ·a), which contracts the same legs.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* T = atomr(ctx, "T", 2);
    auto const* S = atomr(ctx, "S", 2);
    auto const* nested = make_dot(ctx, T, make_dot(ctx, a, S));
    auto const* canon = steps::canonicalize(ctx, nested);
    EXPECT_EQ(infer_rank(canon), std::optional<int>{1});
    auto const* oracle = steps::canonicalize(
        ctx, make_dot(ctx, T, make_dot(ctx, make_transpose(ctx, S), a)));
    EXPECT_TRUE(structural_eq(canon, oracle));
}

// ---- binder sinking over the additive layer (C13) ----------------------

TEST(CanonicalizeNf, BinderDistributesOverSum)
{
    // Σ_i (a_i + b_i) → Σ_i a_i + Σ_i b_i: two terms, not one Paren-wrapped sum
    // (summation is linear; the additive layer stays above the binders).
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    auto const* e = make_explicit_sum(
        ctx,
        i,
        make_sum(
            ctx,
            ivec(ctx, "a", Level::Lower, i),
            ivec(ctx, "b", Level::Lower, i)));
    auto const* nf = canonicalize_nf(ctx, e);
    EXPECT_EQ(nf->terms.size(), 2u);
}

// ---- symbolic division (C13) -------------------------------------------

TEST(CanonicalizeNf, SymbolicDivisionBecomesDiv)
{
    // A/B (non-numeric divisor) → a Div factor; round-trips.
    Context ctx;
    auto const* e =
        make_scalar_div(ctx, atomr(ctx, "A", 2), atomr(ctx, "B", 2));
    auto const* nf = canonicalize_nf(ctx, e);
    ASSERT_EQ(nf->terms.size(), 1u);
    auto const& t = nf->terms[0];
    ASSERT_EQ(t.scalars.size() + t.tensors.size(), 1u);
    auto const* f = t.scalars.empty() ? t.tensors[0] : t.scalars[0];
    EXPECT_TRUE(std::holds_alternative<Div>(f->node));
    EXPECT_TRUE(equal(*nf, *canonicalize_nf(ctx, raise(ctx, *nf))));
}

// ---- unary-of-dyad fence distribution (C13) ----------------------------

TEST(CanonicalizeNf, TransposeOfDyadExpands)
{
    // (a⊗b)^T → b⊗a: a Unary over a ⊗ operand expands; no buried ⊗.
    Context ctx;
    auto const* nf = canonicalize_nf(
        ctx,
        make_transpose(
            ctx, make_tensor_product(ctx, atom(ctx, "a"), atom(ctx, "b"))));
    ASSERT_EQ(nf->terms.size(), 1u);
    ASSERT_EQ(nf->terms[0].tensors.size(), 2u);
    EXPECT_EQ(tname(nf->terms[0], 0), "b"); // b⊗a, positional
    EXPECT_EQ(tname(nf->terms[0], 1), "a");
}
