//! Native C emission directly from the typed IR (no IR -> AST conversion).
use crate::ast::Type;
use crate::ir::{Expr, ExprKind, Module, Statement};
use std::collections::BTreeSet;

fn tag(t: &Type) -> String { match t { Type::F32 => "f32".into(), Type::Vector(n)=>format!("v{n}"),Type::Matrix(r,c)=>format!("m{r}_{c}") } }
fn ctype(t: &Type) -> String { if *t==Type::F32 { "float".into() } else { format!("ml_{}",tag(t)) } }
fn count(t: &Type)->usize { match t { Type::F32=>1,Type::Vector(n)=>*n,Type::Matrix(r,c)=>r*c } }
fn shape(t: &Type)->(usize,usize) { match t { Type::F32=>(1,1),Type::Vector(n)=>(1,*n),Type::Matrix(r,c)=>(*r,*c) } }
fn expr(e:&Expr)->String {
 match &e.kind {
  ExprKind::Constant(n)=>format!("{n:.9}f"),
  ExprKind::Variable(n)=>format!("ml_{n}"),
  ExprKind::Tensor(xs)=>{
   fn flatten(xs:&[Expr],out:&mut Vec<String>){for x in xs {if let ExprKind::Tensor(inner)=&x.kind {flatten(inner,out)} else {out.push(expr(x));}}}
   let mut flat=Vec::new();flatten(xs,&mut flat);
   format!("({}){{{{{}}}}}",ctype(&e.ty),flat.join(", "))
  }
  ExprKind::Neg(x)=>if e.ty==Type::F32 {format!("(-{})",expr(x))} else {format!("ml_neg_{}({})",tag(&e.ty),expr(x))},
  ExprKind::Binary(a,op,b)=>if *op=='@' {format!("ml_matmul_{}_{}({}, {})",tag(&a.ty),tag(&b.ty),expr(a),expr(b))} else if e.ty==Type::F32 {format!("({} {op} {})",expr(a),expr(b))} else {let opname=match op {'+'=>"add",'-'=>"sub",'*'=>"mul",'/'=>"div",_=>unreachable!()};format!("ml_{opname}_{}({}, {})",tag(&e.ty),expr(a),expr(b))},
  ExprKind::Call(n,args)=>{let rendered=args.iter().map(expr).collect::<Vec<_>>().join(", ");match n.as_str(){"dot"=>format!("ml_dot_{}({rendered})",tag(&args[0].ty)),"relu"|"sigmoid"=>format!("ml_{n}_{}({rendered})",tag(&args[0].ty)),_=>format!("ml_{n}({rendered})")}}
 }
}
fn gather(e:&Expr,types:&mut BTreeSet<Type>,ops:&mut BTreeSet<String>) {
 types.insert(e.ty.clone());match &e.kind {
  ExprKind::Tensor(xs)=>for x in xs {gather(x,types,ops)},
  ExprKind::Neg(x)=>{gather(x,types,ops);if x.ty!=Type::F32 {ops.insert(format!("neg:{}",tag(&x.ty)));}},
  ExprKind::Binary(a,op,b)=>{gather(a,types,ops);gather(b,types,ops);if *op=='@' {ops.insert(format!("matmul:{}:{}",tag(&a.ty),tag(&b.ty)));}else if e.ty!=Type::F32 {ops.insert(format!("{}:{}",match op {'+'=>"add",'-'=>"sub",'*'=>"mul",'/'=>"div",_=>unreachable!()},tag(&e.ty)));}},
  ExprKind::Call(n,args)=>{for a in args {gather(a,types,ops);}if ["relu","sigmoid","dot"].contains(&n.as_str()){ops.insert(format!("{n}:{}",tag(&args[0].ty)));}},
  _=>{}
 }
}
fn find_type<'a>(types:&'a BTreeSet<Type>,tag_name:&str)->&'a Type {types.iter().find(|t|tag(t)==tag_name).expect("collected type")}
fn helpers(types:&BTreeSet<Type>,ops:&BTreeSet<String>,printed:&BTreeSet<Type>)->String {
    let mut out=String::new();
    for ty in types {if *ty!=Type::F32 {out.push_str(&format!("typedef struct {{ float data[{}]; }} {};\n",count(ty),ctype(ty)));}}
    out.push('\n');
    for ty in types { if *ty!=Type::F32 {
        let nm=tag(ty);let ct=ctype(ty);let n=count(ty); let (r,c)=shape(ty);
        if printed.contains(ty) {
        out.push_str(&format!("static void ml_print_{nm}({ct} a) {{\n"));
        if matches!(ty,Type::Vector(_)) {out.push_str("  printf(\"[\");\n");}else{out.push_str("  printf(\"[\");\n");}
        out.push_str(&format!("  for (size_t i=0;i<{r};++i) {{\n"));
        if matches!(ty,Type::Matrix(_, _)) {out.push_str("    if(i) printf(\", \"); printf(\"[\");\n");}
        out.push_str(&format!("    for(size_t j=0;j<{c};++j) {{ if(j) printf(\", \"); printf(\"%.9g\",(double)a.data[i*{c}+j]); }}\n"));
        if matches!(ty,Type::Matrix(_, _)) {out.push_str("    printf(\"]\");\n");}
        out.push_str("  }\n  printf(\"]\\n\");\n}\n");
        }
        for operation in ["add","sub","mul","div"] {if ops.contains(&format!("{operation}:{nm}")) {
            let op=match operation {"add"=>"+","sub"=>"-","mul"=>"*",_=>"/"};
            out.push_str(&format!("static {ct} ml_{operation}_{nm}({ct} a,{ct} b) {{ {ct} z; for(size_t i=0;i<{n};++i) z.data[i]=a.data[i]{op}b.data[i]; return z; }}\n"));
        }}
        if ops.contains(&format!("neg:{nm}")) {out.push_str(&format!("static {ct} ml_neg_{nm}({ct} a) {{ {ct} z; for(size_t i=0;i<{n};++i) z.data[i]=-a.data[i]; return z; }}\n"));}
    }}
    for op in ops {
        let parts:Vec<_>=op.split(':').collect();
        match parts.as_slice() {
            &["dot",t]=> {let ty=find_type(types,t);let ct=ctype(ty);let n=count(ty);out.push_str(&format!("static float ml_dot_{t}({ct} a,{ct} b) {{ float s=0; for(size_t i=0;i<{n};++i) s+=a.data[i]*b.data[i]; return s; }}\n"));}
            &["relu",t] | &["sigmoid",t]=> {
                let fn_name=parts[0];let formula=if fn_name=="relu" {"(a.data[i]>0?a.data[i]:0)"}else{"(1.0f/(1.0f+expf(-a.data[i])))"};
                if t=="f32" {let scalar_formula=if fn_name=="relu" {"(a>0?a:0)"}else{"(1.0f/(1.0f+expf(-a)))"};out.push_str(&format!("static float ml_{fn_name}_f32(float a) {{ return {scalar_formula}; }}\n"));}
                else {let ty=find_type(types,t);let ct=ctype(ty);let n=count(ty);out.push_str(&format!("static {ct} ml_{fn_name}_{t}({ct} a) {{ {ct} z; for(size_t i=0;i<{n};++i) z.data[i]={formula}; return z; }}\n"));}
            }
            &["matmul",a,b]=> {
                let l=find_type(types,a);let rt=find_type(types,b);
                let (m,k)=shape(l);let n=match rt {Type::Matrix(_,n)=>*n,Type::Vector(_)=>1,_=>unreachable!()};
                let result=match rt {Type::Vector(_)=>Type::Vector(m),_=>Type::Matrix(m,n)};
                let ct=ctype(&result); let at=ctype(l);let bt=ctype(rt);
                out.push_str(&format!("static {ct} ml_matmul_{a}_{b}({at} a,{bt} b) {{ {ct} z={{0}}; for(size_t i=0;i<{m};++i) for(size_t j=0;j<{n};++j) for(size_t p=0;p<{k};++p) z.data[i*{n}+j]+=a.data[i*{k}+p]*b.data[p*{n}+j]; return z; }}\n"));
            }
            _=>{}
        }
    }
    out
}
pub fn emit_c(module:&Module)->String {
 let mut types=BTreeSet::new();let mut ops=BTreeSet::new();let mut printed=BTreeSet::new();
 for f in &module.functions {
  for (_,t) in &f.params {types.insert(t.clone());}
  if let Some(t)=&f.result {types.insert(t.clone());}
  for stmt in &f.body {match stmt {
   Statement::Let(_,e)|Statement::Return(e)|Statement::Evaluate(e)=>gather(e,&mut types,&mut ops),
   Statement::Print(e)=>{printed.insert(e.ty.clone());gather(e,&mut types,&mut ops);}
  }}
 }
 let mut out=String::from("/* Generated by MLang v0.4 direct typed IR backend. */\n#include <stdio.h>\n#include <stddef.h>\n#include <math.h>\n\n");
 out.push_str(&helpers(&types,&ops,&printed));out.push('\n');
 for f in &module.functions {if f.name=="main" {continue;}let params=f.params.iter().map(|(n,t)|format!("{} ml_{n}",ctype(t))).collect::<Vec<_>>().join(", ");out.push_str(&format!("{} ml_{}({});\n",f.result.as_ref().map(ctype).unwrap_or("void".into()),f.name,if params.is_empty(){"void"}else{&params}));}
 for f in &module.functions {
  let is_main=f.name=="main";let name=if is_main {"main".into()}else{format!("ml_{}",f.name)};
  let ret=if is_main {"int".into()}else{f.result.as_ref().map(ctype).unwrap_or("void".into())};
  let params=f.params.iter().map(|(n,t)|format!("{} ml_{n}",ctype(t))).collect::<Vec<_>>().join(", ");
  out.push_str(&format!("{ret} {name}({}) {{\n",if params.is_empty(){"void"}else{&params}));
  for stmt in &f.body {match stmt {
   Statement::Let(n,e)=>out.push_str(&format!("  {} ml_{n} = {};\n",ctype(&e.ty),expr(e))),
   Statement::Return(e)=>out.push_str(&format!("  return {};\n",expr(e))),
   Statement::Print(e)=>if e.ty==Type::F32 {out.push_str(&format!("  printf(\"%.9g\\n\",(double)({}));\n",expr(e)));}else{out.push_str(&format!("  ml_print_{}({});\n",tag(&e.ty),expr(e)));},
   Statement::Evaluate(e)=>out.push_str(&format!("  (void)({});\n",expr(e))),
  }}
  if is_main {out.push_str("  return 0;\n");}
  out.push_str("}\n\n");
 }
 out
}
