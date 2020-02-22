import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule } from "@angular/forms";
import { HttpClientModule } from "@angular/common/http";
import { Title } from "@angular/platform-browser";

import { MaterialModule } from "./material.module";


@NgModule({
    imports: [
        CommonModule,
        ReactiveFormsModule,
        HttpClientModule,
        MaterialModule,
    ],
    declarations: [
    ],
    exports: [
        ReactiveFormsModule,
        HttpClientModule,
        MaterialModule,
    ],
    entryComponents: [
    ],
    providers: [
        Title,
    ],
})
export class SharedModule { }
