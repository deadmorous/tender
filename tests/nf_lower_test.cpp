#include <tender/nf_lower.hpp>

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

TEST(Encapsulate, UnsupportedNodeThrows)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    // A Sum factor is not yet encapsulable (→ Paren awaits the recursive
    // lower).
    EXPECT_THROW(
        (void)encapsulate(ctx, make_sum(ctx, a, a)), std::invalid_argument);
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

TEST(PlaceFactors, UnknownRankThrows)
{
    Context ctx;
    // A tensor object with no declared rank ⇒ infer_rank is nullopt.
    auto const* x = make_tensor_object(ctx, make_tensor_name("X"));
    EXPECT_THROW(
        (void)place_factors(ctx, ProductParts{Rational{1}, {x}}),
        std::invalid_argument);
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

TEST(LowerTerm, GenuineSumOperandStaysSunkAndAwaitsLower)
{
    // A·(b+c): the sum is NOT distributed (explicit transform only); the sum
    // operand becomes a Paren that still awaits the recursive lower → throws.
    Context ctx;
    auto const* A = atomr(ctx, "A", 2);
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    EXPECT_THROW(
        (void)term1(ctx, make_dot(ctx, A, make_sum(ctx, b, c))),
        std::invalid_argument);
}
