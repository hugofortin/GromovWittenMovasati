__precompile__(true)

module mixedHodge
# simulation of fraction_field by managing denominators separately.

using Oscar
export +,-,*,/, simplify, differentiate, split, integrate, degree, is_homogeneous, weigh, unweigh

#  Basic operations on multivariate polynomials and fraction of those as a pair numerator, denominator
include("fraction_operations.jl")

# weighted monomial iterator
include("monomial_iterate.jl")

export Fg_VF

using LinearAlgebra
using LinearSolve


mutable struct FgenusVectorField
	weights::Vector{Int}              # for variables in Fg=Qg//Fgdenominator  implicit Fgdenominator=(t4-t0^5)^(2g-2)//t5^(3g-3)
	Rs::Vector{Pair{Vector{RingElem},RingElem}}    # coefficients for partial derivatives of Qg in the variables (genus independent)
	Dgs::Vector{<:RingElem} #coefficients for partial derivatives of 1//Fgdenominator (genus dependent)
	Rdiff::Function  # differentiate Fg=Qg//Fgdenominator
	Rsolve::Function  # R4 solving with rhs convolution in R0Fg
	Qg::Vector{<:RingElem} #numerator of Fg along genus g
	R0Fg::Vector{Pair{<:RingElem,<:RingElem}} # R0 Fg as quotient of multivariate polynomial ring elements

	R0squareF1::Pair{<:RingElem,<:RingElem}
	Y::Pair{<:RingElem,<:RingElem}
	F2E::Pair{<:RingElem,<:RingElem}
	F2Y::Pair{<:RingElem,<:RingElem}
end

function Fg_VF(F::Field)
	 R,t= polynomial_ring(F,[Symbol("t"*string(i)) for i=0:6])
	 weights=[3,6,9,12,15,11,8]

	 #coefficients of partial derivatives of multivariate polynomial (genus independent)
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
	     Pair([R(0),-3125t[1]^4-t[4],625*(t[1]^5-t[5]),R(0),R(0),R(0),R(0)],
	          t[6]),
	     Pair([R(0),R(1),R(0),R(0),R(0),R(0),R(0)],
	          R(1)),
	    ]
	 
	 #coefficients of partial derivatives of implicit Fgdenominator (genus dependent)
	 # (t[1]^5-t[5])^(2g-2)*t[6]^(3g-3) 
	 Dgs=[-10t[1]^4*t[6],R(0),R(0),R(0),2t[6],-3(t[1]^5-t[5]),R(0)] # times (g-1)*Qg//(Fgdenominator*(t0^5-t4)*t5)
	 # F1alg=log(t[5]^(-25//12)*(t[5]-t[1]^5)^(5//12)*t[6]^(-1//2)) 
   	 # R02F1alg=1/12*(4687500*t0^10*t6^2 - 185546875*t0^9*t5*t6 -
	 # 904296875*t0^8*t5^2 + 62500*t0^5*t2*t5^2 -
	 # 55000*t0^5*t3*t5*t6 - 9375000*t0^5*t4*t6^2 -
	 # 706250*t0^4*t3*t5^2 + 185546875*t0^4*t4*t5*t6 -
	 # 199218750*t0^3*t4*t5^2 - 112*t3^2*t5^2 -
	 # 62500*t2*t4*t5^2 + 55000*t3*t4*t5*t6 +
	 # 4687500*t4^2*t6^2)/t5^4
 	 # R0F1=mapreduce(i->Rs[1][i]*differentiate(F1alg,i),+,1:length(RS[1]))
	 # println("R0F1 ",(R0F1,Rs[1][2]))
	 R0squareF1=Pair(4687500//12*t[1]^10*t[7]^2 - (185546875//12)*t[1]^9*t[6]*t[7] -
	 		  (904296875//12)*t[1]^8*t[6]^2 + (62500//12)*t[1]^5*t[3]*t[6]^2 -
			  (55000//12)*t[1]^5*t[4]*t[6]*t[7] - (9375000//12)*t[1]^5*t[5]*t[7]^2 -
			  (706250//12)*t[1]^4*t[4]*t[6]^2 + (185546875//12)*t[1]^4*t[5]*t[6]*t[7] -
			  (199218750//12)*t[1]^3*t[5]*t[6]^2 - (112//12)*t[4]^2*t[6]^2-
			  (62500//12)*t[3]*t[5]*t[6]^2 + (55000//12)*t[4]*t[5]*t[6]*t[7] +
			  (4687500//12)*t[5]^2*t[7]^2,
	 		  t[6]^4)
	 # differentiate Fg=Qg/(t4-t0^5)^(2g-2)//t5^(3g-3) #returns quotient of RingElem
	 Rdiff=(Q,indx::Int,genus::Int=0)-> begin 
	 	   if typeof(Q)<:Pair #fraction field genus independent
		      difs=map(i->differentiate(Q,i),1:length(weights))
		      den=mapreduce(ff->ff[2],lcm,difs) # ;init=parent(difs[1][1](1)))
		      num=mapreduce(i->numerator(den//difs[i][2])*difs[i][1]*Rs[indx][1][i],+,1:length(difs))
		      den*=Rs[indx][2]		      
		   elseif typeof(Q)<:RingElem #implicit Qg/Fdenominator genus dependent
	              num=mapreduce(i->Rs[indx][1][i]*((t[1]^5-t[5])*t[6]*differentiate(Q,i)+(genus-1)*Dgs[i]*Q),+,1:length(weights))
		      den=Rs[indx][2]*(t[1]^5-t[5])^(2genus-1)*t[6]^(3genus-2)
		   end
		   simplify(Pair(num,den))
	       end
	 # F1 not algebraic
	 # F1alg=log(t[5]^(-25/12)*(t[5]-t[1]^5)^(5/12)*t[6]^(-1/2)) 
	 Qg=eltype(Dgs)[R(0)] 
	 #R0(F_g) as a quotient (denominator possibly genus dependent)
    	 # R0F1alg=1/12*(3750*t0^5*t6 - 353125*t0^4*t5 - 112*t3*t5 - 3750*t4*t6)/t5^2
	 R0Fg=[Pair((3750//12)*t[7]*(t[1]^5-t[5])-(353125//12)*t[1]^4*t[6]-(112//12)*t[4]*t[6],t[6]^2)]
	 println("check RO2F1 ",Rdiff(R0Fg[1],1)-R0squareF1)
	 #
	 # R4 solving with rhs convolution in R0Fg for given genus 
	 Rsolve=(genus::Int,precision::Real=1.e-15)-> begin
		 		 if genus>length(R0Fg)+1 return(nothing) end
				 #explicit convolution unless R0F denominators are known along genus
	 			 R02=Rdiff(R0Fg[genus-1],1,genus-1)
	 			 println("R0square ",R02)
	 			 conv=mapreduce(r->R0Fg[r]*R0Fg[genus-r],+,1:genus-1)
	 			 println("conv ",typeof(conv)," ",conv)
	 			 # Movasati singular code vs book 1//2 vs -1//(2*5^8)
	 			 scl=R(-1//(2*5^8))
	 			 rhs4=scl*(R02+conv)
	 			 println("R4 F"*string(genus)," convolution rhs=",rhs4)
	 			 lhs4=integrate(rhs4,7)
	 			 lhs4=simplify(Pair(lhs4[1]*Rs[5][2],lhs4[2]*Rs[5][1][7]))
	 			 println("R4 convolution solution upto constant independent of t6 \n",lhs4)
	 			 Rdiff(lhs4,5,2)!=rhs4 && error("check R4 diff lhs vs rhs")
				 #reduce to (t0^5-t4)^(2g-2) t5^(3g-3)
				 implicitden=(t[1]^5-t[5])^(2genus-2)*t[6]^(3genus-3)
				 println("R4 solving",lhs4)
				 # denominator standardized to implicit
				 lhs4=Pair(lhs4[1]*implicitden/lhs4[2],implicitden)
				 println("standardized R4 solution",lhs4)
				 println("==========")
	 			 anomaly=zero(R)
	 			 ambiguity=zero(R)
	 			 if genus==2
	 			    Rdiff(F2Y,5,2)!=rhs4 && error("R4 solving vs F2Y")
	    			    anomaly= -(14//140625)*t[1]^5*t[2]+(19//37500)*t[1]^4*t[3]+(14//28125)*t[1]^3*t[4] + 
	    	      		    	     (14//140625)*t[2]*t[5]+(14//87890625)*t[3]*t[4]
	    			    anomaly*= t[6]^3*(t[1]^5-t[5])
				    
	    			    ambiguity=(56429//36000)*t[1]^12-(2231//1440)*t[1]^7*t[5]-(13//750)*t[1]^2*t[5]^2
	    			    ambiguity*=t[6]^3
	    			    println(anomaly)
	    			    println(ambiguity)
	    			    println("check anomaly and ambiguity ",F2Y[1]-(lhs4[1]+anomaly+ambiguity))
				 end
				 R1lhs4=Rdiff(lhs4[1],2,genus)  # lhs4den implicit
				 R1lhs4[1] != R(0) && error("R1 anomaly")
				 R2lhs4=Rdiff(lhs4[1],3,genus)  # lhs4den implicit
				 R2lhs4=Pair(R2lhs4[1]*implicitden/R2lhs4[2],implicitden)
				 #println("R2lhs4 ",R2lhs4)
				 #println("lhs4 ",lhs4)
				 R3lhs4=Rdiff(lhs4[1],4,genus)  # lhs4den implicit
				 R3lhs4=Pair(R3lhs4[1]*implicitden/R3lhs4[2],implicitden)
				 println("standardized R3lhs4 ",R3lhs4)
				 # anomaly proportional to t5^(3g-3) by R1 anomaly==0 since \\partial t6==0
				 # anomaly proportional to (t0^5-t4) by R2 homogeneity
				 reduced_t0_t4=true
				 targetdeg=69*(genus-1)-weights[6]*(3genus-3)-(reduced_t0_t4 ? weights[1]*5 : 0)
				 # solve R2 and R3 system in t 0:4
				 # R2 anomaly= 2(genus-1) anomaly
				 # R3 anomaly= -R3lhs4[1]
				 sz=mapreduce(p->1,+,weightedmonomial(weights[1:5],targetdeg))
				 println(sz, " unknown coefficients")
				 # monomials to matrix row in unknowns
				 R2rows=Dict{Vector{Int},Vector{<:Rational}}()
				 R3rows=Dict{Vector{Int},Vector{<:Rational}}()
				 if genus==2
				    println("implicit anomaly ",anomaly)
				    r2anomaly=Rdiff(anomaly,3,genus)
				    println("R2 anomaly ",r2anomaly)
				    println("implicit R2 anomaly ",r2anomaly[1]*implicitden/r2anomaly[2])
				    r3anomaly=Rdiff(anomaly,4,genus)
				    println("R3 anomaly ",r3anomaly)
				    println("implicit R3 anomaly ",r3anomaly[1]*implicitden/r3anomaly[2])
				 end
				 for (i,p) in enumerate(weightedmonomial(weights[1:5],targetdeg))
				     fac=prod(t[1:5].^p)*t[6]^(3genus-3)*(reduced_t0_t4 ? (t[1]^5-t[5]) : 1)
				     trm2=Rdiff(fac,3,genus) # implicit denominator
				     # denominator standardized to implicit
				     trm2=Pair(trm2[1]*implicitden/trm2[2],implicitden)
				     println(i,p," R2 ",trm2) 
				     for trm in terms(trm2[1])
				     	 for l=1:length(trm)
					     exps=exponent_vector(trm,l)
					     c=coeff(trm,l)
					     row=get!(R2rows,exps,zeros(Rational{Int},sz+1))
					     row[i]+=c
					 end
				     end
				     #R2 fac=2(g-1)*fac
				     println(i," fac ",fac)
				     for trm in terms(fac)
				     	 for l=1:length(trm)
					     exps=exponent_vector(trm,l)
					     c=coeff(trm,l)
					     row=get!(R2rows,exps,zeros(Rational{Int},sz+1))
					     row[i]+=-2c*(genus-1)
					 end
				     end
				     trm3=Rdiff(fac,4,genus) # implicit denominator
		     		     # denominator standardized to implicit
				     trm3=Pair(trm3[1]*implicitden/trm3[2],implicitden)
				     println(i,p," R3 ",trm3)
				     for trm in terms(trm3[1])
				     	 for l=1:length(trm)
					     exps=exponent_vector(trm,l)
					     c=coeff(trm,l)
					     row=get!(R3rows,exps,zeros(Rational{Int},sz+1))
					     row[i]+=c
					 end
				     end
				     println("================")
				 end
				 # denominator standardized to implicit
				 # R2(anomaly) + R2(lhs4[1]) == 2(genus_1)*(anomaly+lhs4[1])
				 for trm in terms(R2lhs4[1]-2(genus-1)*lhs4[1])
				     for l=1:length(trm)
					 exps=exponent_vector(trm,l)
					 c=coeff(trm,l)
					 row=get(R3rows,exps,[])
				     	 isempty(row) && error("R2 infeasible")
					 row[end]+=c # R2(lhs4[1])-2(genus-1)*lhs4[1] #constant if any
				     end
				 end
				 # R3(anomaly) + R3(lhs4[1]) == 0
				 for trm in terms(R3lhs4[1])
				     for l=1:length(trm)
					 exps=exponent_vector(trm,l)
					 c=coeff(trm,l)
					 row=get(R3rows,exps,[])
				     	 isempty(row) && error("R3 infeasible")
					 row[end]+=c #constant if any
				     end
				 end
				 println(length(R2rows)," R2rows",keys(R2rows))
				 println(length(R3rows)," R3rows",keys(R3rows))
				 #Choose solving method => Rational{BigInt} extremely slow factorization
				 typ=false ? Float64 : Rational{BigInt}
				 rows=length(R2rows)+length(R3rows)
				 A=zero_matrix(F,rows,sz)
				 b=zero_matrix(F,rows,1)
				 i=0
				 for (k,v) in R3rows
				     i+=1
				     A[i,:]=v[1:end-1]
				     b[i]= -v[end]
				 end
				 for (k,v) in R2rows
				     i+=1
				     A[i,:]=v[1:end-1]
				     b[i]= -v[end]
				 end
				 println("A=",A)
				 println("b=",b)
				 if genus==2 #expected solution
				    println("implicit anomaly ",anomaly)
				    #convert anomaly into linear combinations 
				    exp2indx=Dict{Vector{Int},Int}()
				    for (i,p) in enumerate(weightedmonomial(weights[1:5],targetdeg))
				     	fac=prod(t[1:5].^p)*t[6]^(3genus-3)*(reduced_t0_t4 ? (t[1]^5-t[5]) : 1)
					for trm in terms(fac)
				     	    for l=1:length(trm)
					    	exps=exponent_vector(trm,l)
						sgn=sign(coeff(trm,l))
						j=sgn>0 ? i : sgn<0 ? -i : error(trm)
						get!(exp2indx,exps,j)
					    end
					end
				    end
				    println("exp to indx",exp2indx)
				    Sanomaly=zero_matrix(F,sz,1)
				    for trm in terms(anomaly)
				     	for l=1:length(trm)
					    exps=exponent_vector(trm,l)
					    i=get(exp2indx,exps,0)
					    i==0 && error(string(trm))
					    if Sanomaly[abs(i)]==0
					       Sanomaly[abs(i)]=i>0 ? coeff(trm,l) : -coeff(trm,l)
					    end
					    println(exps,i," ",coeff(trm,l))
					end
				    end
				    println("expected solution",Sanomaly)
				    err=A*Sanomaly-b
				    println("Check error of R2 R3 system with anomaly coefficients ",err)
				 end
				 H,U = hnf_with_transform(A)
				 rhs=U*b
				 #backsubstitution
				 R3R2solution=zero_matrix(F,sz,1)
				 println(typeof(H),typeof(R3R2solution))
				 for r=rank(H):-1:1
				     c=r
				     while H[r,c]==0 c+=1 end
				     R3R2solution[c]=rhs[r]-mapreduce(j->H[r,j]*R3R2solution[j],+,c+1:sz;init=F(0))
				     R3R2solution[c]/=H[r,c]
				     println((r,c),R3R2solution[c],"     ",H[r,c])
				 end
				 println("R3R2solution",R3R2solution)
				 println("HNF=",H)
				 println("rhs=",U*b)
				 Rsol=R(0)
				 for (i,p) in enumerate(weightedmonomial(weights[1:5],targetdeg))
				     Rsol+=R3R2solution[i]*prod(t[1:5].^p)*t[6]^(3genus-3)*(reduced_t0_t4 ? (t[1]^5-t[5]) : 1)
				 end
				 println("R2 R3 System solution",Rsol)
				 println("anomaly",anomaly)
				 err=Rsol-anomaly
				 println("Check (computed-expected coefficients) ")
				 for trm in terms(err)
				     for l=1:length(trm)
					 println(exponent_vector(trm,l),Float64(coeff(trm,l)))
					end
				    end
				 println("=================")
				 R2rows,R3rows,A,b,R3R2solution,Rsol
 			 end
	 # data for checking
	 # Yukawa coupling
	 Y= Pair(5^8*(t[5]-t[1]^5)^2,t[6]^3)
	 #Emanuel's F2 and Yamaguch-Yau F2
	 F2E=Pair(-(448//2812500000)*t[6]^3*t[3]*t[4]*t[5]+(60000//2812500000)*t[1]^5*t[7]*t[3]*t[6]^2*t[5]-(30000//2812500000)*t[7]*t[3]*t[6]^2*t[5]^2+(448//2812500000)*t[7]*t[6]^2*t[4]^2*t[5]+(2825000//2812500000)*t[7]*t[6]^2*t[1]^4*t[4]*t[5]-(60000//2812500000)*t[7]^2*t[1]^5*t[6]*t[4]*t[5]+(30000//2812500000)*t[7]^2*t[6]*t[4]*t[5]^2-(937500//2812500000)*t[1]^15*t[7]^3+(4408515625//2812500000)*t[1]^12*t[6]^3+(97500000//2812500000)*t[7]^2*t[1]^14*t[6]+(2812500//2812500000)*t[1]^10*t[7]^3*t[5]-(2812500//2812500000)*t[1]^5*t[7]^3*t[5]^2+(937500//2812500000)*t[7]^3*t[5]^3-(4357421875//2812500000)*t[1]^7*t[5]*t[6]^3-(48750000//2812500000)*t[1]^2*t[5]^2*t[6]^3-(280000//2812500000)*t[1]^10*t[6]^3*t[2]+(1425000//2812500000)*t[1]^9*t[6]^3*t[3]+(1400000//2812500000)*t[1]^8*t[6]^3*t[4]-(4553828125//2812500000)*t[7]*t[6]^2*t[1]^13+(4649453125//2812500000)*t[7]*t[6]^2*t[1]^8*t[5]-(95625000//2812500000)*t[7]*t[6]^2*t[1]^3*t[5]^2-(195000000//2812500000)*t[7]^2*t[1]^9*t[6]*t[5]+(97500000//2812500000)*t[7]^2*t[1]^4*t[6]*t[5]^2-(30000//2812500000)*t[1]^10*t[7]*t[3]*t[6]^2+(560000//2812500000)*t[1]^5*t[6]^3*t[2]*t[5]-(1425000//2812500000)*t[1]^4*t[6]^3*t[3]*t[5]-(280000//2812500000)*t[6]^3*t[2]*t[5]^2+(448//2812500000)*t[1]^5*t[6]^3*t[3]*t[4]-(1400000//2812500000)*t[1]^3*t[6]^3*t[4]*t[5]-(448//2812500000)*t[7]*t[6]^2*t[1]^5*t[4]^2-(2825000//2812500000)*t[7]*t[6]^2*t[1]^9*t[4]+(30000//2812500000)*t[7]^2*t[1]^10*t[6]*t[4],
	 (t[1]^5-t[5])^2*t[6]^3)
	 #Yamaguchi-Yau F2
	 F2Y=Pair((-(1//3000)*t[1]^15*t[7]^3+(13//375)*t[1]^14*t[6]*t[7]^2-(58289//36000)*t[1]^13*t[6]^2*t[7]+(56429//36000)*t[1]^12*t[6]^3-(14//140625)*t[1]^10*t[2]*t[6]^3-(1//93750)*t[1]^10*t[3]*t[6]^2*t[7]+(1//93750)*t[1]^10*t[4]*t[6]*t[7]^2+((1//1000))*t[1]^10*t[5]*t[7]^3+(19//37500)*t[1]^9*t[3]*t[6]^3-(113//112500)*t[1]^9*t[4]*t[6]^2*t[7]-(26//375)*t[1]^9*t[5]*t[6]*t[7]^2+(14//28125)*t[1]^8*t[4]*t[6]^3+(59513//36000)*t[1]^8*t[5]*t[6]^2*t[7]-(2231//1440)*t[1]^7*t[5]*t[6]^3+(28//140625)*t[1]^5*t[2]*t[5]*t[6]^3+(14//87890625)*t[1]^5*t[3]*t[4]*t[6]^3+(1//46875)*t[1]^5*t[3]*t[5]*t[6]^2*t[7]-(14//87890625)*t[1]^5*t[4]^2*t[6]^2*t[7]-(1//46875)*t[1]^5*t[4]*t[5]*t[6]*t[7]^2-(1//1000)*t[1]^5*t[5]^2*t[7]^3-(19//37500)*t[1]^4*t[3]*t[5]*t[6]^3+(113//112500)*t[1]^4*t[4]*t[5]*t[6]^2*t[7]+(13//375)*t[1]^4*t[5]^2*t[6]*t[7]^2-(14//28125)*t[1]^3*t[4]*t[5]*t[6]^3-(17//500)*t[1]^3*t[5]^2*t[6]^2*t[7]-(13//750)*t[1]^2*t[5]^2*t[6]^3-(14//140625)*t[2]*t[5]^2*t[6]^3-(14//87890625)*t[3]*t[4]*t[5]*t[6]^3-(1//93750)*t[3]*t[5]^2*t[6]^2*t[7]+(14//87890625)*t[4]^2*t[5]*t[6]^2*t[7]+(1//93750)*t[4]*t[5]^2*t[6]*t[7]^2+(1//3000)*t[5]^3*t[7]^3),
	 (t[1]^5-t[5])^2* t[6]^3)
	Fg=FgenusVectorField(weights,Rs,Dgs,Rdiff,Rsolve,Qg,R0Fg,R0squareF1,Y,F2E,F2Y)
	push!(Fg.Qg,Fg.F2Y[1])
	push!(Fg.R0Fg,Rdiff(Fg.F2Y,1,2))
	Fg
end


end # module mixedHodge

