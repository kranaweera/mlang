use crate::ast::{Expr, Program, Stmt, Type};
use crate::semantic::infer;
use std::collections::{BTreeSet, HashMap};

type Sigs = HashMap<String,(Vec<Type>,Option<Type>)>;
fn tag(t: &Type) -> String { match t { Type::F32 => "f32".into(), Type::Vector(n)=>format!("v{n}"),Type::Matrix(r,c)=>format!("m{r}_{c}") } }
fn ctype(t: &Type) -> String { if *t==Type::F32 { "float".into() } else { format!("ml_{}",tag(t)) } }
fn count(t: &Type)->usize { match t { Type::F32=>1,Type::Vector(n)=>*n,Type::Matrix(r,c)=>r*c } }
fn shape(t: &Type)->(usize,usize) { match t { Type::F32=>(1,1),Type::Vector(n)=>(1,*n),Type::Matrix(r,c)=>(*r,*c) } }
fn expr(e: &Expr, vars: &HashMap<String,Type>, sigs: &Sigs) -> String {
    match e {
        Expr::Number(n)=>format!("{n:.9}f"), Expr::Var(name)=>format!("ml_{name}"),
        Expr::Vector(xs)=> {
            let ty=infer(e,vars,sigs).expect("validated AST");
            let mut flat=Vec::new();
            fn flatten(x:&Expr,out:&mut Vec<Expr>) { match x { Expr::Vector(ys)=>for y in ys {flatten(y,out)}, _=>out.push(x.clone()) } }
            for x in xs { flatten(x,&mut flat); }
            format!("({}){{{{{}}}}}",ctype(&ty),flat.iter().map(|x|expr(x,vars,sigs)).collect::<Vec<_>>().join(", "))
        }
        Expr::UnaryMinus(x)=> { let ty=infer(x,vars,sigs).unwrap(); if ty==Type::F32 {format!("(-{})",expr(x,vars,sigs))} else {format!("ml_neg_{}({})",tag(&ty),expr(x,vars,sigs))} }
        Expr::Binary(a,op,b)=> {
            let lt=infer(a,vars,sigs).unwrap(); let rt=infer(b,vars,sigs).unwrap();
            if *op=='@' { format!("ml_matmul_{}_{}({}, {})",tag(&lt),tag(&rt),expr(a,vars,sigs),expr(b,vars,sigs)) }
            else if lt==Type::F32 {format!("({} {op} {})",expr(a,vars,sigs),expr(b,vars,sigs))}
            else { let opname=match op {'+'=>"add",'-'=>"sub",'*'=>"mul",'/'=>"div",_=>unreachable!()};format!("ml_{opname}_{}({}, {})",tag(&lt),expr(a,vars,sigs),expr(b,vars,sigs)) }
        }
        Expr::Call(name,args)=> {
            let rendered=args.iter().map(|x|expr(x,vars,sigs)).collect::<Vec<_>>().join(", ");
            match name.as_str() {
                "dot"=>format!("ml_dot_{}({rendered})",tag(&infer(&args[0],vars,sigs).unwrap())),
                "relu"|"sigmoid"=>format!("ml_{name}_{}({rendered})",tag(&infer(&args[0],vars,sigs).unwrap())),
                _=>format!("ml_{name}({rendered})"),
            }
        }
    }
}
fn gather(e:&Expr,vars:&HashMap<String,Type>,sigs:&Sigs,types:&mut BTreeSet<Type>,ops:&mut BTreeSet<String>) {
    let t=infer(e,vars,sigs).expect("validated AST"); types.insert(t.clone());
    match e {
        Expr::Vector(xs)=>for x in xs {gather(x,vars,sigs,types,ops)},
        Expr::UnaryMinus(x)=> {gather(x,vars,sigs,types,ops);let xt=infer(x,vars,sigs).unwrap();if xt!=Type::F32 {ops.insert(format!("neg:{}",tag(&xt)));} },
        Expr::Binary(a,op,b)=> {
            gather(a,vars,sigs,types,ops);gather(b,vars,sigs,types,ops);
            let lt=infer(a,vars,sigs).unwrap();let rt=infer(b,vars,sigs).unwrap();
            if *op=='@' {ops.insert(format!("matmul:{}:{}",tag(&lt),tag(&rt)));}
            else if lt!=Type::F32 {ops.insert(format!("{}:{}",match op {'+'=>"add",'-'=>"sub",'*'=>"mul",'/'=>"div",_=>unreachable!()},tag(&lt)));}
        }
        Expr::Call(name,args)=> {
            for x in args {gather(x,vars,sigs,types,ops);}
            if ["relu","sigmoid","dot"].contains(&name.as_str()) {let t=infer(&args[0],vars,sigs).unwrap();ops.insert(format!("{name}:{}",tag(&t)));}
        }
        _=>{}
    }
}
fn find_type<'a>(types:&'a BTreeSet<Type>,tag_name:&str)->&'a Type {types.iter().find(|t|tag(t)==tag_name).expect("collected type")}
fn helpers(types:&BTreeSet<Type>,ops:&BTreeSet<String>)->String {
    let mut out=String::new();
    for ty in types {if *ty!=Type::F32 {out.push_str(&format!("typedef struct {{ float data[{}]; }} {};\n",count(ty),ctype(ty)));}}
    out.push('\n');
    for ty in types { if *ty!=Type::F32 {
        let nm=tag(ty);let ct=ctype(ty);let n=count(ty); let (r,c)=shape(ty);
        out.push_str(&format!("static void ml_print_{nm}({ct} a) {{\n"));
        if matches!(ty,Type::Vector(_)) {out.push_str("  printf(\"[\");\n");}else{out.push_str("  printf(\"[\");\n");}
        out.push_str(&format!("  for (size_t i=0;i<{r};++i) {{\n"));
        if matches!(ty,Type::Matrix(_, _)) {out.push_str("    if(i) printf(\", \"); printf(\"[\");\n");}
        out.push_str(&format!("    for(size_t j=0;j<{c};++j) {{ if(j) printf(\", \"); printf(\"%.9g\",(double)a.data[i*{c}+j]); }}\n"));
        if matches!(ty,Type::Matrix(_, _)) {out.push_str("    printf(\"]\");\n");}
        out.push_str("  }\n  printf(\"]\\n\");\n}\n");
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
pub fn emit_c(module:&crate::ir::Module)->String {
    let program = &crate::ir::to_ast(module);
    let sigs:Sigs=program.functions.iter().map(|f|(f.name.clone(),(f.params.iter().map(|(_,t)|t.clone()).collect(),f.result.clone()))).collect();
    let mut types=BTreeSet::new();let mut ops=BTreeSet::new();
    for f in &program.functions {
        let mut vars:HashMap<String,Type>=f.params.iter().cloned().collect();
        for (_,t) in &f.params {types.insert(t.clone());}
        if let Some(t)=&f.result {types.insert(t.clone());}
        for stmt in &f.body {let e=match stmt {Stmt::Let(_,e)|Stmt::Return(e)|Stmt::Expr(e)=>e};
            if let Stmt::Expr(Expr::Call(name,args))=stmt {if name=="print" {gather(&args[0],&vars,&sigs,&mut types,&mut ops);continue;}}
            gather(e,&vars,&sigs,&mut types,&mut ops);
            if let Stmt::Let(name,e)=stmt {vars.insert(name.clone(),infer(e,&vars,&sigs).unwrap());}
        }
    }
    let mut out=String::from("/* Generated by MLang v0.3 typed IR backend. */\n#include <stdio.h>\n#include <stddef.h>\n#include <math.h>\n\n");
    out.push_str(&helpers(&types,&ops));out.push('\n');
    for f in &program.functions {if f.name=="main" {continue;}let params=f.params.iter().map(|(n,t)|format!("{} ml_{n}",ctype(t))).collect::<Vec<_>>().join(", ");out.push_str(&format!("{} ml_{}({});\n",f.result.as_ref().map(ctype).unwrap_or("void".into()),f.name,if params.is_empty(){"void"}else{&params}));}
    for f in &program.functions {
        let main=f.name=="main";let name=if main {"main".into()}else{format!("ml_{}",f.name)};
        let ret=if main {"int".into()}else{f.result.as_ref().map(ctype).unwrap_or("void".into())};
        let params=f.params.iter().map(|(n,t)|format!("{} ml_{n}",ctype(t))).collect::<Vec<_>>().join(", ");
        out.push_str(&format!("{ret} {name}({}) {{\n",if params.is_empty(){"void"}else{&params}));
        let mut vars:HashMap<String,Type>=f.params.iter().cloned().collect();
        for stmt in &f.body {match stmt {
            Stmt::Let(n,e)=>{let ty=infer(e,&vars,&sigs).unwrap();out.push_str(&format!("  {} ml_{n} = {};\n",ctype(&ty),expr(e,&vars,&sigs)));vars.insert(n.clone(),ty);}
            Stmt::Return(e)=>out.push_str(&format!("  return {};\n",expr(e,&vars,&sigs))),
            Stmt::Expr(Expr::Call(n,args)) if n=="print"=>{let e=&args[0];let ty=infer(e,&vars,&sigs).unwrap();if ty==Type::F32 {out.push_str(&format!("  printf(\"%.9g\\n\",(double)({}));\n",expr(e,&vars,&sigs)));}else{out.push_str(&format!("  ml_print_{}({});\n",tag(&ty),expr(e,&vars,&sigs)));}}
            Stmt::Expr(e)=>out.push_str(&format!("  (void)({});\n",expr(e,&vars,&sigs))),
        }}
        if main {out.push_str("  return 0;\n");}
        out.push_str("}\n\n");
    }
    out
}
