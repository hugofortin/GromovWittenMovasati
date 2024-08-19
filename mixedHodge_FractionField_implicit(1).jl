__precompile__(true)

module mixedHodge
# simulation of fraction_field by managing denominators separately.

using Oscar
import Oscar: degree, is_homogeneous, simplify
export Fg_VF, weigh, unweigh, simplify, differentiate, integrate, degree, is_homogeneous, R0Fg

function simplify(ff::Pair{<:RingElem,<:RingElem})
	 q=gcd(ff[1],ff[2])
	 q==parent(ff[1])(1) ? ff : Pair(ff[1]/q,ff[2]/q)
end


function differentiate(pol::RingElem, var::Int)
	 PR=parent(pol)
	 z=zero(PR)
	 res=z
	 for t in terms(pol)
     	     res+=mapreduce(l->(exps=exponent_vector(t,l);exps[var]-=1;exps[var]< 0 ? z : coeff(t,l)*(exps[var]+1)*mapreduce(i->PR[i]^exps[i],*,1:length(exps))),+,1:length(t))
	 end
	 res
end

"""
differentiate quotient
"""
function differentiate(ff::Pair{<:RingElem,<:RingElem},var::Int)
	 difnum=differentiate(ff[1],var)
	 difden=differentiate(ff[2],var)
	 difden==0 ? Pair(difnum,ff[2]) : Pair(difnum*ff[2]-ff[1]*difden,ff[2]^2)
end

function integrate(pol::RingElem, var::Int)
	 PR=parent(pol)
	 z=zero(PR)
	 res=z
	 for t in terms(pol)
     	     res+=mapreduce(l->(exps=exponent_vector(t,l);exps[var]+=1;coeff(t,l)//exps[var]*mapreduce(i->PR[i]^exps[i],*,1:length(exps))),+,1:length(t))
	 end
	 res
end

"""
integrate quotient
"""
function integrate(ff::Pair{<:RingElem,<:RingElem},var::Int)
	 difden=differentiate(ff[2],var)
	 if difden==0
	    Pair(integrate(ff[1],var),ff[2])
	 else #by parts
	    error("integrate by parts to do")
	 end
end


function degree(pol::RingElem, weights::Vector{Int})
	 degs=Int[]
	 for t in terms(pol)
     	     map(l->(exps=exponent_vector(t,l);push!(degs,mapreduce(i->exps[i]*weights[i],+,1:length(exps)))),1:length(t))
	 end
	 degs
end
degree(ff::Pair{<:RingElem,<:RingElem}, weights::Vector{Int})=(degree(ff[1],weights),degree(ff[2],weights))

is_homogeneous(pol::RingElem, weights::Vector{Int})= length(unique(degree(pol,weights)))==1
is_homogeneous(ff::Pair{<:RingElem,<:RingElem}, weights::Vector{Int})=is_homogeneous(ff[1],weights)&&is_homogeneous(ff[2],weights)

"""
introduce weights in exponents
"""
function weigh(pol::RingElem, weights::Vector{Int})
	 PR=parent(pol)
	 z=zero(PR)
	 res=z
	 for t in terms(pol)
     	     res+=mapreduce(l->(exps=exponent_vector(t,l);exps.*=weights[1:length(exps)];coeff(t,l)*mapreduce(i->PR[i]^exps[i],*,1:length(exps))),+,1:length(t))
	 end
	 res
end
weigh(ff::Pair{<:RingElem,<:RingElem}, weights::Vector{Int})=Pair(weigh(ff[1],weights),weigh(ff[2],weights))

"""
remove weights in exponents if feasible otherwise raise an error
"""
function unweigh(pol::RingElem, weights::Vector{Int})
	 PR=parent(pol)
	 z=zero(PR)
	 res=z
	 for t in terms(pol)
     	     res+=mapreduce(l->(exps=exponent_vector(t,l);exps./=weights[1:length(exps)];coeff(t,l)*mapreduce(i->PR[i]^exps[i],*,1:length(exps))),+,1:length(t))
	 end
	 res
end
unweigh(ff::Pair{<:RingElem,<:RingElem}, weights::Vector{Int})=Pair(unweigh(ff[1],weights),unweigh(ff[2],weights))



mutable struct FgenusVectorField
	weights::Vector{Int}              # for variables in Fg=Qg//Fgdenominator  implicit Fgdenominator=(t4-t0^5)^(2g-2)//t5^(3g-3)
	Rs::Vector{Pair{Vector{RingElem},RingElem}}    # coefficients for partial derivatives of Qg in the variables (genus independent)
	Dgs::Vector{<:RingElem} #coefficients for partial derivatives of 1//Fgdenominator (genus dependent)
	Rdiff::Function  # differentiate Fg=Qg//Fgdenominator
	Qg::Vector{<:RingElem} #numerator of Fg along genus g
	R0Fg::Vector{Pair{RingElem,RingElem}} # R0 Fg as quotient of multivariate polynomial ring elements

	R0squareF1::Pair{<:RingElem,<:RingElem}
	Y::Pair{<:RingElem,<:RingElem}
	F2E::Pair{<:RingElem,<:RingElem}
	F2Y::Pair{<:RingElem,<:RingElem}
end

function Fg_VF(F::Field)
	 R,t= polynomial_ring(F,[Symbol("t"*string(i)) for i=0:6])
	 weights=[3,6,9,12,15,11,8]
	 Rs=[
	     Pair([625*(t[1]^5-t[5])+3125t[1]^5+t[1]*t[4],
		   -390625t[1]*(t[1]^5-t[5])+3125t[1]^4*t[2]+t[2]*t[4],
		   -5859375t[1]^2*(t[1]^5-t[5])-625t[2]*(t[1]^5-t[5])+6250t[1]^4*t[3]+2t[3]*t[4],
		   -9765625t[1]^3*(t[1]^5-t[5])-625t[3]*(t[1]^5-t[5])+9375t[1]^4*t[4]+3t[4]^2,
		   15625t[1]^4*t[5]+5t[4]*t[5],
		   -625t[7]*(t[1]^5-t[5])+9375t[1]^4*t[6]+2t[4]*t[6],
		   9375t[1]^4*t[7]− 3125t[1]^3*t[6]-2t[3]*t[6]+3t[4]*t[7]],
		  t[6]),
	     Pair([R(0),R(0),R(0),R(0),R(0),t[6],t[7]],
	          R(1)),
	     Pair([t[1],2t[2],3t[3],4t[4],5t[5],3t[6],2t[7]],
		  R(1)),
	     Pair([R(0),
	           -5t[1]^4*t[7]+5t[1]^3*t[6]+(1//625)*t[3]*t[6]-(1//625)*t[4]*t[7],
		   t[7]*(t[1]^5-t[5]),
		   t[6]*(t[1]^5-t[5]),
		   R(0),R(0),R(0)],
		  t[1]^5-t[5]),
	     Pair([R(0),R(0),R(0),R(0),R(0),R(0),625*(t[1]^5-t[5])],
	          t[6]),
	     Pair([R(0),R(0),R(0),R(0),R(0),R(0),-3125t[1]^4-t[4]],
	          t[6]),
	     Pair([R(0),R(1),R(0),R(0),R(0),R(0),R(0)],
	          R(1)),
	    ]
	 Dgs=[-10t[1]^4*t[6],R(0),R(0),R(0),2t[6],-3(t[1]^5-t[5]),R(0)] # times (g-1)*Qg//(Fgdenominator*(t0^5-t4)*t5)
println(eltype(Dgs),eltype(Rs))	
	 R0squareF1=Pair(-390625*t[1]^10*t[7]^2 + 185546875//12*t[1]^9*t[6]*t[7] + 904296875//12*t[1]^8*t[6]^2 - 15625//3*t[1]^5*t[3]*t[6]^2 + 13750//3*t[1]^5*t[4]*t[6]*t[7] + 781250*t[1]^5*t[5]*t[7]^2 + 353125//6*t[1]^4*t[4]*t[6]^2 - 185546875//12*t[1]^4*t[5]*t[6]*t[7] + 33203125//2*t[1]^3*t[5]*t[6]^2 + 15625//3*t[3]*t[5]*t[6]^2 + 28//3*t[4]^2*t[6]^2 - 13750//3*t[4]*t[5]*t[6]*t[7] - 390625*t[5]^2*t[7]^2,
	 t[6]^4)
	 # differentiate Fg=Qg/(t4-t0^5)^(2g-2)//t5^(3g-3) #returns quotient of RingElem
	 Rdiff=(Q,indx::Int,g::Int=0)-> begin 
	 	   if typeof(Q)<:Pair #fraction field genus independent
		      difs=map(i->differentiate(Q,i),1:length(weights))
		      den=mapreduce(ff->ff[2],lcm,difs) # ;init=parent(difs[1][1](1)))
		      num=mapreduce(i->numerator(den//difs[i][2])*difs[i][1]*Rs[indx][1][i],+,1:length(difs))
		      den*=Rs[indx][2]		      
		   elseif typeof(Q)<:RingElem #implicit Qg/Fdenominator genus dependent
	              num=mapreduce(i->Rs[indx][1][i]*((t[1]^5-t[5])*t[6]*differentiate(Q,i)+(g-1)*Dgs[i]*Q),+,1:length(weights))
		      den=Rs[indx][2]*(t[1]^5-t[5])^(2g-1)*t[6]^(3g-2)
		   end
		   simplify(Pair(num,den))
	       end
	 Qg=eltype(Dgs)[R(0)] # F1 not algebraic
	 #R0(F_g) as a quotient (denominator possibly genus dependent)
	 R0Fg=[Pair(-(3750//12)*t[7]*(t[1]^5-t[5])+(353125//12)*t[1]^4*t[6]+(112//12)*t[4]*t[6],t[6]^2)]
	 # data for checking
	 # Yukawa coupling
	 Y= Pair(5^8*(t[5]-t[1]^5)^2,t[6]^3)
	 #Emanuel's F2 and Yamaguch-Yau F2
	 F2E=Pair(-(448//2812500000)*t[6]^3*t[3]*t[4]*t[5]+(60000//2812500000)*t[1]^5*t[7]*t[3]*t[6]^2*t[5]-(30000//2812500000)*t[7]*t[3]*t[6]^2*t[5]^2+(448//2812500000)*t[7]*t[6]^2*t[4]^2*t[5]+(2825000//2812500000)*t[7]*t[6]^2*t[1]^4*t[4]*t[5]-(60000//2812500000)*t[7]^2*t[1]^5*t[6]*t[4]*t[5]+(30000//2812500000)*t[7]^2*t[6]*t[4]*t[5]^2-(937500//2812500000)*t[1]^15*t[7]^3+(4408515625//2812500000)*t[1]^12*t[6]^3+(97500000//2812500000)*t[7]^2*t[1]^14*t[6]+(2812500//2812500000)*t[1]^10*t[7]^3*t[5]-(2812500//2812500000)*t[1]^5*t[7]^3*t[5]^2+(937500//2812500000)*t[7]^3*t[5]^3-(4357421875//2812500000)*t[1]^7*t[5]*t[6]^3-(48750000//2812500000)*t[1]^2*t[5]^2*t[6]^3-(280000//2812500000)*t[1]^10*t[6]^3*t[2]+(1425000//2812500000)*t[1]^9*t[6]^3*t[3]+(1400000//2812500000)*t[1]^8*t[6]^3*t[4]-(4553828125//2812500000)*t[7]*t[6]^2*t[1]^13+(4649453125//2812500000)*t[7]*t[6]^2*t[1]^8*t[5]-(95625000//2812500000)*t[7]*t[6]^2*t[1]^3*t[5]^2-(195000000//2812500000)*t[7]^2*t[1]^9*t[6]*t[5]+(97500000//2812500000)*t[7]^2*t[1]^4*t[6]*t[5]^2-(30000//2812500000)*t[1]^10*t[7]*t[3]*t[6]^2+(560000//2812500000)*t[1]^5*t[6]^3*t[2]*t[5]-(1425000//2812500000)*t[1]^4*t[6]^3*t[3]*t[5]-(280000//2812500000)*t[6]^3*t[2]*t[5]^2+(448//2812500000)*t[1]^5*t[6]^3*t[3]*t[4]-(1400000//2812500000)*t[1]^3*t[6]^3*t[4]*t[5]-(448//2812500000)*t[7]*t[6]^2*t[1]^5*t[4]^2-(2825000//2812500000)*t[7]*t[6]^2*t[1]^9*t[4]+(30000//2812500000)*t[7]^2*t[1]^10*t[6]*t[4],
	 (t[1]^5-t[5])^2*t[6]^3)
	 #Yamaguchi-Yau F2
	 F2Y=Pair((-(1//3000)*t[1]^15*t[7]^3+(13//375)*t[1]^14*t[6]*t[7]^2-(58289//36000)*t[1]^13*t[6]^2*t[7]+(56429//36000)*t[1]^12*t[6]^3-(14//140625)*t[1]^10*t[2]*t[6]^3-(1//93750)*t[1]^10*t[3]*t[6]^2*t[7]+(1//93750)*t[1]^10*t[4]*t[6]*t[7]^2+((1//1000))*t[1]^10*t[5]*t[7]^3+(19//37500)*t[1]^9*t[3]*t[6]^3-(113//112500)*t[1]^9*t[4]*t[6]^2*t[7]-(26//375)*t[1]^9*t[5]*t[6]*t[7]^2+(14//28125)*t[1]^8*t[4]*t[6]^3+(59513//36000)*t[1]^8*t[5]*t[6]^2*t[7]-(2231//1440)*t[1]^7*t[5]*t[6]^3+(28//140625)*t[1]^5*t[2]*t[5]*t[6]^3+(14//87890625)*t[1]^5*t[3]*t[4]*t[6]^3+(1//46875)*t[1]^5*t[3]*t[5]*t[6]^2*t[7]-(14//87890625)*t[1]^5*t[4]^2*t[6]^2*t[7]-(1//46875)*t[1]^5*t[4]*t[5]*t[6]*t[7]^2-(1//1000)*t[1]^5*t[5]^2*t[7]^3-(19//37500)*t[1]^4*t[3]*t[5]*t[6]^3+(113//112500)*t[1]^4*t[4]*t[5]*t[6]^2*t[7]+(13//375)*t[1]^4*t[5]^2*t[6]*t[7]^2-(14//28125)*t[1]^3*t[4]*t[5]*t[6]^3-(17//500)*t[1]^3*t[5]^2*t[6]^2*t[7]-(13//750)*t[1]^2*t[5]^2*t[6]^3-(14//140625)*t[2]*t[5]^2*t[6]^3-(14//87890625)*t[3]*t[4]*t[5]*t[6]^3-(1//93750)*t[3]*t[5]^2*t[6]^2*t[7]+(14//87890625)*t[4]^2*t[5]*t[6]^2*t[7]+(1//93750)*t[4]*t[5]^2*t[6]*t[7]^2+(1//3000)*t[5]^3*t[7]^3),
	 (t[1]^5-t[5])^2* t[6]^3)
	println(typeof(F2E),typeof(F2Y))
	FgenusVectorField(weights,Rs,Dgs,Rdiff,Qg,R0Fg,R0squareF1,Y,F2E,F2Y)
end


function R0Fg(Fg::FgenusVectorField,genus::Int)
	 wght=ones(Int,length(Fg.weights))
	 FF=parent(Fg.F2Y)
	 sclt5=-FF[7]*FF[6]//625*(FF[1]^5-FF[5])
	 for g=(length(Fg.R0Fgs)+1):genus
	     sclr4=(FF[5]-FF[1])^(2g-2)*FF[6]^(3g-3)*FF[6]//625//(FF[1]^5-FF[5])
	     rhsr4=Fg.Rs[1](Fg.R0Fgs[g-1])+sum([Fg.R0Fgs[r]*Fg.R0Fgs[g-r] for r=1:g-1])
	     rhsr4//=2 
	     println("rhsr4",rhsr4)
	     println(degree(rhsr4,Fg.weights))
	     # R4 Fg= rhs  integrate in t6
	     Qg_t6=sclr4*integrate(rhsr4,7)
	     println("Qg_t6=")
	     println(numerator(Qg_t6))
	     println("//",denominator(Qg_t6))
	     println(degree(Qg_t6,Fg.weights))
	     # R1 Fg=0 integrate in t5
	     rhsr1= -FF[7]//625//(FF[1]^5-FF[5])*rhsr4
	     println("rhsr1=",rhsr1)
	     println(degree(rhsr1,Fg.weights))
	     println(numerator(rhsr4))
	     println("//",denominator(rhsr4))
	     numt5=integrate(numerator(rhsr4),6)
	     dent5=integrate(denominator(rhsr4),6)
	     println("Qg_t5=",numt5)
	     println("//",dent5)
	 end
end		    
# weighted monomial iterator
include("monomial_iterate.jl")

end # module mixedHodge
